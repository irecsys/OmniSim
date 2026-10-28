"""
LLM fallbacks are visible. Offline: fake clients; errors are injected at a call
site by replacing utils.llm_calls._invoke (the single place that calls a client).

1. each fallback returns its (domain-neutral, non-leaking) value, writes exactly one WARNING naming the site and
   the error type, and is counted; secrets in error messages never reach the log or the summary
2. full runs (run_simulation, toy data): <FALLBACK> line, run summary, exit code, .aborted transcripts, the
   startup check, profile caching, one thread running several conversations, several threads
3. tools: the <FALLBACK> line does not change what metrics / NLP metrics / corpus comparison read; judge and
   evaluator do not show it to the LLM and record a 'fallbacks' column; summaries and .aborted files are not
   read as conversations
"""
import glob
import hashlib
import json
import logging
import os
import re
import types

import numpy as np
import pandas as pd
import pytest
import requests
import yaml

import run as run_py
import utils.llm_calls as llm_calls
import utils.quick_start as qs
import utils.user_profile_builder as upb
import utils.utils as uu
from utils.configurator import Config
from utils.dialogue_files import split_fallback_line

FALLBACK_RE = re.compile(r'^System: <FALLBACK> fallbacks=(\d+) sites=([a-z_]+:\d+(?:,[a-z_]+:\d+)*) '
                         r'errors=([A-Za-z_]+:\d+(?:,[A-Za-z_]+:\d+)*)$')
DOMAIN_WORDS = re.compile(r'(?i)\b(cinematic|cinema|films?|movies?|watch\w*|popcorn|screen|reel|flick|wear|outfit|'
                          r'clothes|clothing|dress|fashion|shop\w*)\b')
REPLIES = ['Something with a clear personality would be good.', 'I want a darker tone.',
           '["genre", "year"]', 'Sure, here is a calm one.', '{"value": "1990"}', 'drama, ghosts, eerie',
           'That is not what I meant.', 'A cozy pick for tonight!']


def _h(text):
    """Deterministic across processes (hash() of a str is not)."""
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)


class FakeChat:
    def __init__(self, content=None):
        self.chat = types.SimpleNamespace(completions=self)
        self.content = content

    def create(self, **kwargs):
        content = self.content
        if content is None:
            content = REPLIES[_h(json.dumps(kwargs.get('messages'), default=str)) % len(REPLIES)]
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeEmbeddings:
    def __init__(self, dim=8):
        self.embeddings = self
        self.dim = dim

    def create(self, model=None, input=None, **kwargs):
        rng = np.random.default_rng(_h(str(input)) % (2 ** 32))
        v = rng.normal(size=self.dim)
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=(v / np.linalg.norm(v)).tolist())])


SERVER_ERROR = '500 Server Error: Internal Server Error for url: https://example.invalid/infer'


def http_error(text='403 Client Error: Forbidden for url: https://example.invalid/infer'):
    return requests.HTTPError(text)


@pytest.fixture
def config():
    return Config(['configs/imdb/imdb.yaml'])


@pytest.fixture
def inject(monkeypatch):
    """inject(rule): rule is {site: exc} or a function (site) -> exc or None; matching calls raise."""
    def setup(rule):
        real = llm_calls._invoke

        def invoke(site, fn, kwargs):
            exc = rule(site) if callable(rule) else rule.get(site)
            if exc is not None:
                raise exc
            return real(site, fn, kwargs)
        monkeypatch.setattr(llm_calls, '_invoke', invoke)
    return setup


@pytest.fixture
def record(monkeypatch):
    """A fresh conversation record for the test thread."""
    rec = llm_calls.Record()
    monkeypatch.setattr(llm_calls._local, 'record', rec, raising=False)
    return rec


def fallback_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.getMessage().startswith('LLM call failed, using fallback text')]


# ── 1. the fallbacks one by one ───────────────────────────────────────────────
ROW = pd.Series({'genres': 'Horror', 'overview': 'A haunted farmhouse hides a secret. The family flees. Then more.',
                 'release_year': 1999, 'original_language': 'en'})
RECS = [{'imdb_id': 'tt1', 'title': 'Some Film', 'genres': 'Drama'}]

CASES = [
    # id, site, call(config, client) -> value, expected fallback value
    ('expand_query_for_search', 'expand_query_for_search', lambda c, k: uu.expand_query_for_search(c, k, 'a dark story'), 'a dark story'),
    ('generate_chit_chat-prefix', 'generate_chit_chat', lambda c, k: uu.generate_chit_chat(c, k, 'bot', context='pre_recommendation'), ''),
    ('generate_chit_chat-transition', 'generate_chit_chat', lambda c, k: uu.generate_chit_chat(c, k, 'bot', context='transition'), ''),
    ('generate_chit_chat-greeting', 'generate_chit_chat', lambda c, k: uu.generate_chit_chat(c, k, 'bot', context='greeting'),
     uu.NEUTRAL_GREETING),
    ('generate_chit_chat-closing', 'generate_chit_chat', lambda c, k: uu.generate_chit_chat(c, k, 'bot', context='closing'),
     uu.NEUTRAL_CLOSING),
    ('generate_rejection_explanation', 'generate_rejection_explanation', lambda c, k: uu.generate_rejection_explanation(c, k, ROW, RECS),
     "None of these match what I'm looking for."),
    ('generate_recommendation_explanation', 'generate_recommendation_explanation', lambda c, k: uu.generate_recommendation_explanation(c, k, RECS), ''),
    ('generate_bot_closing_failed', 'generate_bot_closing_failed', lambda c, k: uu.generate_bot_closing_failed(c, k),
     "Didn't get there today — but I'm sure you'll find it."),
    ('rephrase_phrase', 'rephrase_phrase', lambda c, k: uu.rephrase_phrase(c, k, 'What are you looking for?'),
     'What are you looking for?'),
    ('rephrase_phrase-greeting', 'rephrase_phrase',
     lambda c, k: uu.rephrase_phrase(c, k, 'What kind of movie are you after?', fallback_text=uu.NEUTRAL_GREETING),
     uu.NEUTRAL_GREETING),
    ('profile_demographic', 'profile_demographic', lambda c, k: upb.generate_demographic_summary(k, c, pd.Series({'age': 30})), ''),
    ('profile_likes_long', 'profile_likes_long', lambda c, k: upb._llm(k, c, 'summarise', site='profile_likes_long'), ''),
    ('profile_likes_short', 'profile_likes_short', lambda c, k: upb._llm(k, c, 'summarise', site='profile_likes_short'), ''),
]


@pytest.mark.parametrize('site, call, expected', [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_fallback_value_warning_and_count(config, inject, record, caplog, site, call, expected):
    inject({site: http_error()})
    with caplog.at_level(logging.WARNING):
        assert call(config, FakeChat()) == expected
    msgs = fallback_warnings(caplog)
    assert len(msgs) == 1
    assert f'site={site}' in msgs[0] and 'error=HTTPError' in msgs[0]
    assert record.fallbacks == {site: 1}
    assert record.errors == {(site, 'HTTPError'): 1}
    assert record.calls[site] == 1


def test_whole_turn_fallbacks_are_domain_neutral():
    for text in (uu.NEUTRAL_GREETING, uu.NEUTRAL_CLOSING):
        assert text and not DOMAIN_WORDS.search(text)


@pytest.mark.parametrize('dataset', ['imdb', 'hm'])
def test_free_user_reply_fallback_is_neutral(inject, record, caplog, dataset):
    """In generate_user_responses_free_mode a failed short reply falls through to the detailed reply; the
    detailed reply no longer copies the target's details (v1 leaked the answer) but uses the
    domain-neutral free_user_fallback_prompts."""
    config = Config([f'configs/{dataset}/{dataset}.yaml'])
    config['short_reply_ratio'] = 1.0
    details_col = config['col_details']
    row = pd.Series({details_col: 'Grim Farmhouse Story. A haunted farmhouse hides a secret. The family flees.',
                     config['col_category']: 'Horror'})
    inject({'free_user_short_reply': http_error(), 'free_user_details_reply': http_error()})
    pool = uu.load_prompts(config)['free_user_fallback_prompts']
    with caplog.at_level(logging.WARNING):
        msg = uu.generate_user_responses_free_mode(config, FakeChat(), row)
    assert msg in [p.replace('"', '') for p in pool]
    assert 'farmhouse' not in msg.lower() and 'grim' not in msg.lower()
    assert all(not DOMAIN_WORDS.search(p) for p in pool)
    assert record.fallbacks == {'free_user_short_reply': 1, 'free_user_details_reply': 1}
    assert len(fallback_warnings(caplog)) == 2


def test_no_error_no_warning_calls_counted(config, record, caplog):
    with caplog.at_level(logging.WARNING):
        assert uu.rephrase_phrase(config, FakeChat('Hi there.'), 'Hello.') == 'Hi there.'
    assert fallback_warnings(caplog) == []
    assert record.calls == {'rephrase_phrase': 1} and not record.errors and not record.fallbacks


def test_uncaught_site_is_counted_and_reraised(config, inject, record):
    exc = http_error()
    inject({'extract_item_from_title': exc})
    with pytest.raises(requests.HTTPError) as ei:
        uu.extract_item_from_title(config, FakeChat(), 'Some Film', 'Drama')
    assert ei.value is exc and exc.omnisim_llm_site == 'extract_item_from_title'
    assert record.errors == {('extract_item_from_title', 'HTTPError'): 1} and not record.fallbacks


def test_unregistered_site_is_refused():
    with pytest.raises(ValueError, match='unregistered'):
        llm_calls.chat(FakeChat(), 'no_such_site', model='m', messages=[])


def test_secrets_are_masked(config, inject, record, caplog, monkeypatch):
    monkeypatch.setenv('THETA_KEY', 'theta-secret-value-123')
    monkeypatch.setenv('AZURE_KEY', 'azure-secret-value-456')
    text = ('401 Unauthorized: Authorization: Bearer abc.def.ghi api-key=XYZ789secret '
            'sk-proj-ABCDEF123456 key theta-secret-value-123 and azure-secret-value-456')
    inject({'rephrase_phrase': http_error(text)})
    with caplog.at_level(logging.WARNING):
        uu.rephrase_phrase(config, FakeChat(), 'Hello.')
    msg = fallback_warnings(caplog)[0]
    for secret in SECRETS:
        assert secret not in msg
    assert '***' in msg and '401 Unauthorized' in msg


SECRETS = ('abc.def.ghi', 'XYZ789secret', 'sk-proj-ABCDEF123456', 'theta-secret-value-123', 'azure-secret-value-456')


def test_sanitize_truncates():
    assert len(llm_calls.sanitize('x' * 1000)) == llm_calls.MAX_MESSAGE


def test_theta_retries_counted_apart(monkeypatch, record):
    """The Theta client retries 409 / 429 / 503 inside one logical call: one call, retries counted by status."""
    statuses = iter([429, 503, 200])

    class Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            pass

        def json(self):
            return {'body': {'infer_requests': [{'output': {'message': 'ok'}}]}}
    monkeypatch.setattr(uu._requests, 'post', lambda *a, **k: Resp(next(statuses)))
    monkeypatch.setattr('time.sleep', lambda s: None)
    out = llm_calls.chat(uu.ThetaEdgeCloudClient('k', 'slug'), 'rephrase_phrase', model='m', messages=[])
    assert out.choices[0].message.content == 'ok'
    assert record.calls == {'rephrase_phrase': 1}
    assert record.retries == {('rephrase_phrase', 429): 1, ('rephrase_phrase', 503): 1}


def test_run_py_exit_code():
    assert run_py.exit_code({'exit_code': 3}) == 3
    assert run_py.exit_code({'exit_code': 0}) == 0
    assert run_py.exit_code(None) == 0


# ── 2. full runs on toy data ──────────────────────────────────────────────────
GENRES = ['Drama', 'Comedy', 'Horror', 'Action', 'Romance']


def toy(tmp_path, n_items=30, ratings=(5, 4, 5, 2), pairs=None, extra=None, profiles=False):
    ds = tmp_path / 'data' / 'toy'
    ds.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for k in range(n_items):
        v = rng.normal(size=8)
        rows.append({'item_id': f'tt{k}', 'title': f'Movie {k}', 'category': f'{GENRES[k % 5]}, {GENRES[(k + 2) % 5]}',
                     'details': f'A {GENRES[k % 5].lower()} story number {k}. About {["love", "war", "ghosts"][k % 3]}.',
                     'release_year': 1980 + k, 'embedding_vector': str((v / np.linalg.norm(v)).tolist())})
    pd.DataFrame(rows).to_csv(ds / 'items.csv', index=False)
    pd.DataFrame({'user_id': ['u1', 'u2', 'u3'], 'age': [30, 40, 50]}).to_csv(ds / 'users.csv', index=False)
    pd.DataFrame({'user_id': ['u1', 'u1', 'u2', 'u3'], 'item_id': ['tt1', 'tt2', 'tt3', 'tt4'],
                  'rating': list(ratings)}).to_csv(ds / 'interactions.csv', index=False)
    pairs = pairs or [('u1', 'tt5'), ('u2', 'tt11')]
    pd.DataFrame({'user_id': [p[0] for p in pairs], 'item_id': [p[1] for p in pairs]}).to_csv(
        tmp_path / 'pairs.csv', index=False)
    cfg = {'data_path': str(tmp_path / 'data') + '/', 'dataset': 'toy', 'file_items': 'items.csv',
           'file_users': 'users.csv', 'file_interactions': 'interactions.csv' if profiles else None,
           'index_dir': str(tmp_path / 'index' / 'toy'), 'precomputed_embedding_col': 'embedding_vector',
           'col_userid': 'user_id', 'col_itemid': 'item_id', 'col_rating': 'rating', 'col_title': 'title',
           'col_category': 'category', 'col_details': 'details', 'embedding_fields': ['title', 'category', 'details'],
           'item_attributes': {'category': 'genre', 'release_year': 'year'}, 'default_attributes': ['genre'],
           'bm25_details_field': 'bm25_details', 'role_user': 'movie seeker', 'role_bot': 'movie recommender',
           'log_path_outputs': str(tmp_path / 'logs') + '/', 'outputs_folder': str(tmp_path / 'chats') + '/',
           'input_pairs_file': str(tmp_path / 'pairs.csv'), 'input_items_file': None, 'input_users_file': None,
           'guestuser_randomitem_records': None, 'chats_per_entry': 1, 'num_workers': 1,
           'embeddings_provider': 'openai', 'embeddings_model': 'fake-embed'}
    cfg.update(extra or {})
    path = tmp_path / 'toy.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return str(path)


def run_toy(tmp_path, monkeypatch, mode='free', chat=None, rebuild_profiles=False, **toy_kw):
    cfg_file = toy(tmp_path, profiles=rebuild_profiles, **toy_kw)
    clients = lambda config: (chat or FakeChat(), FakeEmbeddings())
    monkeypatch.setattr(qs, 'get_openai_clients', clients)
    monkeypatch.setattr(upb, 'get_openai_clients', clients)
    if not rebuild_profiles:
        monkeypatch.setattr(qs, '_needs_profile_rebuild', lambda config, data_path: False)
    summary = qs.run_simulation([cfg_file], mode_override=mode)
    files = sorted(glob.glob(str(tmp_path / 'chats' / '**' / '*.*'), recursive=True))
    return summary, files


def finals(files):
    return [f for f in files if f.endswith('.txt') and '-tmp-' not in f]


def summary_file(tmp_path, summary, mode):
    path = tmp_path / 'chats' / 'toy' / mode / f"run_summary_{summary['run_timestamp']}.json"
    assert path.exists()
    with open(path, encoding='utf-8') as f:
        assert json.load(f) == json.loads(json.dumps(summary, default=str))
    return path


def file_fallbacks(path):
    lines = open(path, encoding='utf-8').read().splitlines()
    fb = [l for l in lines if l.startswith('System: <FALLBACK>')]
    assert len(fb) <= 1
    if fb:
        assert lines[-1] == fb[0] and FALLBACK_RE.match(fb[0]), lines[-3:]
        assert lines[-2].startswith(('System: <END>', 'System: <FAILED>'))
        return int(FALLBACK_RE.match(fb[0]).group(1))
    return 0


@pytest.mark.parametrize('mode', ['free', 'static', 'adaptive'])
def test_run_without_errors(tmp_path, monkeypatch, caplog, mode):
    with caplog.at_level(logging.INFO):
        summary, files = run_toy(tmp_path, monkeypatch, mode)
    assert summary['exit_code'] == 0 and summary['status'] == 'completed'
    assert len(finals(files)) == 2 and all(file_fallbacks(f) == 0 for f in finals(files))
    assert not [f for f in files if f.endswith('.aborted') or '-tmp-' in f]
    c = summary['conversations']
    assert (c['dispatched'], c['saved'], c['aborted'], c['dropped'], c['skipped']) == (2, 2, 0, 0, 0)
    assert c['succeeded'] + c['failed'] == 2
    assert summary['fallbacks']['total'] == 0 and summary['llm_errors']['total'] == 0
    assert summary['calls']['by_site']['startup_check'] == 1 and summary['calls']['by_site']['query_embedding'] >= 2
    assert set(summary['calls']['by_site']) <= set(llm_calls.SITES)
    assert summary['retries']['total'] == 0
    assert (summary['chat_provider'], summary['chat_model'], summary['embeddings_provider'],
            summary['embeddings_model'], summary['on_llm_error']) == (
        'azure', 'gpt-4o-mini', 'openai', 'fake-embed', 'fallback')
    assert summary['startup_check'] == {'ok': None} or summary['startup_check'].get('ok') is not False
    summary_file(tmp_path, summary, mode)
    assert any('Run summary written to' in r.getMessage() for r in caplog.records)


FALLBACK_SITES = {
    'free': ['expand_query_for_search', 'generate_chit_chat', 'rephrase_phrase', 'free_user_details_reply',
             'generate_rejection_explanation', 'generate_bot_closing_failed'],
    'static': ['expand_query_for_search', 'attribute_question'],
    'adaptive': ['attribute_question', 'generate_recommendation_explanation'],
}


@pytest.mark.parametrize('mode, site', [(m, s) for m, sites in FALLBACK_SITES.items() for s in sites])
def test_run_with_fallbacks(tmp_path, monkeypatch, inject, mode, site):
    inject({site: http_error()})
    summary, files = run_toy(tmp_path, monkeypatch, mode)
    per_file = {f: file_fallbacks(f) for f in finals(files)}
    assert len(per_file) == 2
    total = sum(per_file.values())
    if total == 0:
        pytest.skip(f'{site} was not reached in this toy run')
    assert summary['fallbacks']['total'] == total
    assert summary['fallbacks']['by_site'] == {site: total}
    assert summary['fallbacks']['conversations_with_fallback'] == sum(n > 0 for n in per_file.values())
    assert summary['exit_code'] == 3
    assert {a['file'] for a in summary['affected']} == {
        os.path.relpath(f, tmp_path / 'chats') for f, n in per_file.items() if n}
    summary_file(tmp_path, summary, mode)


@pytest.mark.parametrize('mode, site, file_kept', [
    ('free', 'query_embedding', True), ('free', 'extract_item_from_title', True),
    ('static', 'extract_dynamic_description', True), ('adaptive', 'extract_candidate_attributes', True),
    ('static', 'rephrase_attribute', True)])
def test_uncaught_error_keeps_aborted_transcript(tmp_path, monkeypatch, inject, mode, site, file_kept):
    inject({site: http_error(SERVER_ERROR)})        # not 403: a Theta 403 stops dispatching (test_empty_filters_and_dispatch_stop)
    summary, files = run_toy(tmp_path, monkeypatch, mode)
    aborted = [f for f in files if f.endswith('.aborted')]
    assert len(aborted) == 2 and not finals(files) and not [f for f in files if f.endswith('.txt')]
    for f in aborted:
        last = open(f, encoding='utf-8').read().splitlines()[-1]
        assert last == f'System: <ABORTED> site={site} error=HTTPError on_llm_error=fallback'
    c = summary['conversations']
    assert (c['aborted'], c['saved'], c['dropped']) == (2, 0, 0)
    assert summary['exit_code'] == 3
    assert {a['status'] for a in summary['affected']} == {'aborted'}
    assert all(a['first_error']['site'] == site for a in summary['affected'])


def test_error_before_the_transcript_exists(tmp_path, monkeypatch, inject):
    """simulator._get_user_embeddings (profile embedding) runs before the tmp file is created: aborted, no file."""
    cfg_file = toy(tmp_path)
    monkeypatch.setattr(qs, 'get_openai_clients', lambda config: (FakeChat(), FakeEmbeddings()))
    monkeypatch.setattr(qs, '_needs_profile_rebuild', lambda config, data_path: False)
    pd.DataFrame([{'userid': 'u1', 'user_demographic': 'x', 'user_likes_long': 'likes ghosts',
                   'user_likes_short': 'likes ghosts'}]).to_csv(tmp_path / 'data' / 'toy' / 'user_profiles.csv', index=False)
    inject({'profile_embedding': http_error(SERVER_ERROR)})
    summary = qs.run_simulation([cfg_file], mode_override='free')
    by_user = {a['user_id']: a for a in summary['affected']}
    assert by_user['u1']['status'] == 'aborted' and by_user['u1']['file'] is None
    assert summary['conversations']['aborted'] == 1 and summary['conversations']['saved'] == 1
    assert summary['exit_code'] == 3


def test_startup_check_failure_stops_the_run(tmp_path, monkeypatch, inject, caplog):
    inject({'startup_check': http_error('401 Unauthorized api-key=XYZ789secret')})
    summary, files = run_toy(tmp_path, monkeypatch, 'free')
    assert not [f for f in files if not f.endswith('.json')]
    assert summary['status'] == 'aborted' and summary['aborted_at'] == 'startup_check'
    assert summary['startup_check']['ok'] is False and summary['startup_check']['error'] == 'HTTPError'
    assert summary['conversations']['dispatched'] == 0 and summary['exit_code'] == 3
    text = summary_file(tmp_path, summary, 'free').read_text(encoding='utf-8')
    assert 'XYZ789secret' not in text and 'XYZ789secret' not in caplog.text


def test_one_thread_fallback_then_clean_conversation(tmp_path, monkeypatch, inject):
    """The per-conversation record starts fresh: a conversation after one with fallbacks, in the same worker
    thread, has no <FALLBACK> line, and the summary counts only the first."""
    first_item = {'value': None}

    def rule(site):
        rec = llm_calls.current()
        if rec is None:
            return None
        if first_item['value'] is None:
            first_item['value'] = rec.item_id
        return http_error() if site == 'expand_query_for_search' and rec.item_id == first_item['value'] else None
    inject(rule)
    summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=[('u1', 'tt5'), ('u2', 'tt11')])
    per_item = {os.path.basename(f).split('-')[1]: file_fallbacks(f) for f in finals(files)}
    assert per_item[first_item['value']] > 0
    assert [n for item, n in per_item.items() if item != first_item['value']] == [0]
    assert summary['fallbacks']['conversations_with_fallback'] == 1
    assert summary['fallbacks']['total'] == per_item[first_item['value']]


def test_threads_do_not_mix_records(tmp_path, monkeypatch, inject):
    """4 workers, 12 conversations; errors are injected only in conversations of some items (decided from the
    worker thread's own record): exactly those files have a <FALLBACK> line; the totals match."""
    pairs = [(f'u{1 + k % 3}', f'tt{k}') for k in range(12)]
    failing = {'tt0', 'tt3', 'tt4', 'tt9', 'tt10'}

    def rule(site):
        rec = llm_calls.current()
        if rec is not None and str(rec.item_id) in failing and site in ('expand_query_for_search', 'rephrase_phrase'):
            return http_error()
        return None
    inject(rule)
    summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=pairs, extra={'num_workers': 4})
    per_item = {os.path.basename(f).split('-')[1]: file_fallbacks(f) for f in finals(files)}
    assert len(per_item) == 12
    assert {i for i, n in per_item.items() if n} == failing
    assert summary['fallbacks']['total'] == sum(per_item.values())
    assert summary['fallbacks']['conversations_with_fallback'] == len(failing)
    assert summary['conversations']['saved'] == 12


def test_profiles_with_failed_calls_are_not_cached(tmp_path, monkeypatch, inject, caplog):
    def rule(site):
        return http_error() if site == 'profile_likes_long' else None
    inject(rule)
    with caplog.at_level(logging.WARNING):
        summary, files = run_toy(tmp_path, monkeypatch, 'free', rebuild_profiles=True)
    ds = tmp_path / 'data' / 'toy'
    assert not (ds / 'user_profiles.csv').exists() and not (ds / '.profile_hash.json').exists()
    failed = summary['profiles']['failed_users']
    assert {f['user_id'] for f in failed} == {'u1', 'u2'} and {f['site'] for f in failed} == {'profile_likes_long'}
    assert summary['profiles']['cache_written'] is False and summary['exit_code'] == 3
    assert len(finals(files)) == 2                                     # the run goes on with the profiles in memory
    assert any('were not updated' in r.getMessage() for r in caplog.records)


def test_profiles_without_failures_are_cached(tmp_path, monkeypatch):
    summary, files = run_toy(tmp_path, monkeypatch, 'free', rebuild_profiles=True)
    ds = tmp_path / 'data' / 'toy'
    assert (ds / 'user_profiles.csv').exists() and (ds / '.profile_hash.json').exists()
    assert summary['profiles'] == {'rebuilt': True, 'cache_written': True, 'failed_users': []}
    assert summary['exit_code'] == 0


def test_round_limit_is_a_normal_failure(tmp_path, monkeypatch):
    """Conversations stopped by max_total_rounds are saved as failed and do not change the exit code."""
    summary, files = run_toy(tmp_path, monkeypatch, 'free', extra={'max_total_rounds': 1, 'max_questions': 0,
                                                                 'max_rec_attempts': 99},
                             pairs=[('u1', 'tt5'), ('u2', 'tt11'), ('u3', 'tt17')])
    c = summary['conversations']
    assert c['stopped_by_max_total_rounds'] == c['failed'] >= 1
    assert c['saved'] == 3 and summary['exit_code'] == 0


# ── 3. tools ──────────────────────────────────────────────────────────────────
DIALOGUE = ('Bot: What are you after?\nUser: A haunted farmhouse story.\nBot: How about "Movie 5" (tt5)?\n'
            'User: Yes!\nSystem: <END> Session ended successfully. The target item \'Movie 5\' (tt5) was accepted by the user.\n')
FB_LINE = 'System: <FALLBACK> fallbacks=2 sites=expand_query_for_search:1,rephrase_phrase:1 errors=HTTPError:2\n'


def two_folders(tmp_path):
    a, b = tmp_path / 'plain', tmp_path / 'with_line'
    for d, text in ((a, DIALOGUE), (b, DIALOGUE + FB_LINE)):
        d.mkdir()
        (d / 'u1-tt5-3-1-1-20261025120059.txt').write_text(text, encoding='utf-8')
        (d / 'run_summary_20261025120059.json').write_text('{"exit_code": 3}', encoding='utf-8')
        (d / 'user_item_pairs-tt6-u2-tmp-20261025120100-0a1b2c3d.aborted').write_text(
            'Bot: Hi\nSystem: <ABORTED> site=query_embedding error=HTTPError on_llm_error=fallback\n', encoding='utf-8')
    return str(a), str(b)


def test_split_fallback_line():
    assert split_fallback_line(DIALOGUE + FB_LINE) == (DIALOGUE, 2)
    assert split_fallback_line(DIALOGUE) == (DIALOGUE, 0)


def test_metric_tools_ignore_the_line_summaries_and_aborted(tmp_path):
    import scripts.compute_nlp_metrics as cnm
    import utils.compare_metrics as cm
    import utils.metrics as um
    a, b = two_folders(tmp_path)
    assert um.parse_utterances(os.path.join(a, 'u1-tt5-3-1-1-20261025120059.txt')) == \
        um.parse_utterances(os.path.join(b, 'u1-tt5-3-1-1-20261025120059.txt'))
    assert cnm.load_utterances(a) == cnm.load_utterances(b)
    assert cm._load_utterances(a) == cm._load_utterances(b)
    assert cm.item_mention_entropy(a) == cm.item_mention_entropy(b)
    assert cm._conv_stats(a) == cm._conv_stats(b) == {'success_rate': 1.0, 'avg_turns': 3.0}
    assert len(um.compute_metrics(a)) == len(um.compute_metrics(b)) == 1


def test_corpus_comparison_classify_ignores_the_line():
    import legacy_es_equivalence.compare_dialogue_corpora as cdc
    assert cdc.classify(DIALOGUE) == cdc.classify(DIALOGUE + FB_LINE) == ('success', 'Movie 5')
    assert cdc.bot_turns(DIALOGUE) == cdc.bot_turns(DIALOGUE + FB_LINE)


class RecordingJudge:
    def __init__(self, reply):
        self.chat = types.SimpleNamespace(completions=self)
        self.seen, self.reply = [], reply

    def create(self, **kwargs):
        self.seen.append(kwargs['messages'][-1]['content'])
        msg = types.SimpleNamespace(content=self.reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_judge_hides_the_line_and_records_fallbacks(tmp_path):
    import utils.judge as judge
    reply = json.dumps({k: {'score': 4, 'rationale': 'ok'} for k in
                        ('Language_Fluency', 'Conversational_Quality', 'Content_Quality')} |
                       {'overall_score': 4, 'Overall_Assessment': 'ok'})
    for folder, n in zip(two_folders(tmp_path), (0, 2)):
        client = RecordingJudge(reply)
        df = judge.evaluate_folder(folder, client, 'm', workers=1)
        assert len(client.seen) == 1 and '<FALLBACK>' not in client.seen[0] and 'haunted farmhouse' in client.seen[0]
        assert list(df['fallbacks']) == [n]


def test_evaluator_hides_the_line_and_records_fallbacks(tmp_path, monkeypatch, config):
    import utils.evaluator as ev
    reply = json.dumps({'fluency': {'score': 4, 'reason': 'ok'}, 'informativeness': {'score': 4, 'reason': 'ok'},
                        'coherence': {'score': 4, 'reason': 'ok'}})
    for folder, n in zip(two_folders(tmp_path), (0, 2)):
        client = RecordingJudge(reply)
        monkeypatch.setattr(ev, 'get_openai_clients', lambda cfg: (client, None))
        out = tmp_path / f'ev_{n}.csv'
        ev.evaluate_folder(config, folder, str(out))
        assert len(client.seen) == 1 and '<FALLBACK>' not in client.seen[0]
        assert list(pd.read_csv(out)['fallbacks']) == [n]


# ── 4. on_llm_error: fail ─────────────────────────────────────────────────────
def test_invalid_on_llm_error_is_a_config_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='on_llm_error'):
        run_toy(tmp_path, monkeypatch, 'free', extra={'on_llm_error': 'retry'})


@pytest.mark.parametrize('mode, site', [('free', 'rephrase_phrase'), ('free', 'expand_query_for_search'),
                                        ('free', 'free_user_details_reply'), ('static', 'attribute_question'),
                                        ('adaptive', 'attribute_question'), ('free', 'query_embedding')])
def test_fail_mode_stops_and_skips_the_rest(tmp_path, monkeypatch, inject, mode, site):
    """One worker, three pairs: the first conversation stops at the first error and is kept as .aborted; the
    other two are not started (skipped); no <FALLBACK> text was used."""
    inject({site: http_error()})
    summary, files = run_toy(tmp_path, monkeypatch, mode, extra={'on_llm_error': 'fail'},
                             pairs=[('u1', 'tt5'), ('u2', 'tt11'), ('u3', 'tt17')])
    aborted = [f for f in files if f.endswith('.aborted')]
    assert len(aborted) == 1 and not [f for f in files if f.endswith('.txt')]
    last = open(aborted[0], encoding='utf-8').read().splitlines()[-1]
    assert last == f'System: <ABORTED> site={site} error=HTTPError on_llm_error=fail'
    c = summary['conversations']
    assert (c['dispatched'], c['aborted'], c['skipped'], c['saved']) == (3, 1, 2, 0)
    assert summary['skip_reasons'] == {'on_llm_error=fail': 2}
    assert summary['fallbacks']['total'] == 0 and summary['on_llm_error'] == 'fail'
    assert summary['exit_code'] == 3
    summary_file(tmp_path, summary, mode)


def test_fail_mode_with_several_threads(tmp_path, monkeypatch, inject):
    """4 workers: the conversation of tt0 stops at its first call; the conversations already running finish
    normally, the ones not started yet are skipped."""
    def rule(site):
        rec = llm_calls.current()
        return http_error() if rec is not None and str(rec.item_id) == 'tt0' else None
    inject(rule)
    pairs = [(f'u{1 + k % 3}', f'tt{k}') for k in range(10)]
    summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=pairs,
                             extra={'on_llm_error': 'fail', 'num_workers': 4})
    c = summary['conversations']
    assert c['aborted'] == 1 and c['dispatched'] == 10
    assert c['saved'] + c['skipped'] == 9 and c['skipped'] >= 6 and c['saved'] <= 3
    assert len(finals(files)) == c['saved'] and all(file_fallbacks(f) == 0 for f in finals(files))
    assert summary['exit_code'] == 3


def test_fail_mode_profile_error_stops_the_run(tmp_path, monkeypatch, inject):
    inject({'profile_likes_long': http_error()})
    summary, files = run_toy(tmp_path, monkeypatch, 'free', rebuild_profiles=True, extra={'on_llm_error': 'fail'})
    ds = tmp_path / 'data' / 'toy'
    assert not (ds / 'user_profiles.csv').exists() and not (ds / '.profile_hash.json').exists()
    assert summary['status'] == 'aborted' and summary['aborted_at'] == 'profiles'
    assert summary['profiles']['stopped_at']['site'] == 'profile_likes_long'
    assert summary['profiles']['stopped_at']['error'] == 'HTTPError'
    assert summary['conversations']['dispatched'] == 0 and summary['exit_code'] == 3
    assert not [f for f in files if not f.endswith('.json')]


def test_fail_mode_startup_check(tmp_path, monkeypatch, inject):
    inject({'startup_check': http_error()})
    summary, files = run_toy(tmp_path, monkeypatch, 'free', extra={'on_llm_error': 'fail'})
    assert summary['aborted_at'] == 'startup_check' and summary['exit_code'] == 3


def test_all_four_strategies_run_inside_the_conversation_scope(tmp_path, monkeypatch):
    """items, item_list, user_item_pairs and user_guest: every conversation is merged into the summary."""
    pd.DataFrame({'item_id': ['tt1', 'tt2']}).to_csv(tmp_path / 'items_in.csv', index=False)
    pd.DataFrame({'user_id': ['u1', 'u2']}).to_csv(tmp_path / 'users_in.csv', index=False)
    summary, files = run_toy(tmp_path, monkeypatch, 'free', extra={
        'guestuser_randomitem_records': 2, 'input_items_file': str(tmp_path / 'items_in.csv'),
        'input_users_file': str(tmp_path / 'users_in.csv'), 'chats_per_entry': 1})
    assert set(summary['per_strategy']) == {'items', 'item_list', 'user_item_pairs', 'user_guest'}
    for strategy, counts in summary['per_strategy'].items():
        assert counts['dispatched'] == counts.get('saved', 0) + counts.get('aborted', 0) + counts.get('dropped', 0), strategy
    assert summary['conversations']['saved'] == len(finals(files)) == summary['conversations']['dispatched']
    assert summary['calls']['by_site']['query_embedding'] >= summary['conversations']['saved']
