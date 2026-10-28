"""
Visibility only, no change to retrieval or dialogue.

1. A hard filter that on its own leaves no candidate for a structural reason — year_range on a dataset without a
   release_year column, require_any_genre that no category contains — logs one WARNING per conversation and
   reason, and the run summary counts conversations and empty retrievals; the conversation files are the same as
   with the notification switched off.
2. A ThetaEdgeCloud HTTP 403 that stops a conversation stops dispatching new ones (as the daily rate limit does);
   the summary says why. Counting, fallbacks, exit code and on_llm_error: fail are unchanged.
Offline: toy data and fake clients from test_llm_fallback.
"""
import glob
import logging
import os

import numpy as np
import pandas as pd
import pytest
import requests

import utils.llm_calls as llm_calls
import utils.quick_start as qs
import utils.user_profile_builder as upb
from test_llm_fallback import FakeChat, FakeEmbeddings, finals, inject, run_toy, summary_file, toy  # noqa: F401

WARN = 'Hard filter left no candidates'
REASONS = ('year_range_without_column', 'require_any_genre_no_match')
PAIRS = [('u1', 'tt5'), ('u2', 'tt11')]
LIMITS = {'max_total_rounds': 4, 'max_questions': 2}       # 4 retrievals per conversation, all empty


def hm_like(tmp_path, monkeypatch, reply, notify=True):
    """H&M-shaped toy catalogue: no release_year column, fashion categories without any genre word. Every chat
    call returns `reply`, so the user's turns carry the constraint from the first turn on."""
    cfg_file = toy(tmp_path, pairs=PAIRS, extra={**LIMITS, 'item_attributes': {'category': 'category'},
                                                 'role_user': 'fashion shopper', 'role_bot': 'fashion assistant'})
    rng = np.random.default_rng(0)
    rows = []
    for k in range(30):
        v = rng.normal(size=8)
        rows.append({'item_id': f'tt{k}', 'title': f'Item {k}',
                     'category': ['Women Tops Basic', 'Men Shoes Dressed', 'Kids Outerwear'][k % 3],
                     'details': f'A soft jersey piece number {k}.',
                     'embedding_vector': str((v / np.linalg.norm(v)).tolist())})
    pd.DataFrame(rows).to_csv(tmp_path / 'data' / 'toy' / 'items.csv', index=False)
    clients = lambda config: (FakeChat(reply), FakeEmbeddings())
    monkeypatch.setattr(qs, 'get_openai_clients', clients)
    monkeypatch.setattr(upb, 'get_openai_clients', clients)
    monkeypatch.setattr(qs, '_needs_profile_rebuild', lambda config, data_path: False)
    if not notify:
        monkeypatch.setattr(llm_calls, 'note_empty_filter', lambda *a, **k: None)
    summary = qs.run_simulation([cfg_file], mode_override='free')
    files = sorted(glob.glob(str(tmp_path / 'chats' / '**' / '*.txt'), recursive=True))
    return summary, files


def transcripts(files):
    """{(user, item): text}; the final file name ends with the run timestamp, which differs between runs."""
    return {tuple(os.path.basename(f).split('-')[:2]): open(f, 'rb').read() for f in finals(files)}


def warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and WARN in r.getMessage()]


@pytest.mark.parametrize('reply, reason, value, column', [
    ('Something I loved back in 1999.', 'year_range_without_column', '(1997, 2001)', "'release_year'"),
    ('Something in Crime, please.', 'require_any_genre_no_match', "['Crime']", "'category'"),
])
def test_hm_like_empty_filter_is_logged_once_and_counted(tmp_path, monkeypatch, caplog, reply, reason, value, column):
    with caplog.at_level(logging.WARNING):
        summary, files = hm_like(tmp_path / 'on', monkeypatch, reply)
    found = warnings(caplog)
    assert len(found) == len(PAIRS)                                    # one per conversation, not one per round
    for user, item in PAIRS:
        msg = next(r.getMessage() for r in found if f'user {user}, item {item}' in r.getMessage())
        assert value in msg and column in msg and 'dataset toy' in msg
    counts = summary['empty_after_filters']
    assert counts[reason] == {'conversations': 2, 'empty_rounds': 2 * LIMITS['max_total_rounds']}
    assert all(counts[r] == {'conversations': 0, 'empty_rounds': 0} for r in REASONS if r != reason)
    assert summary['calls']['by_site']['query_embedding'] == 2 * LIMITS['max_total_rounds']
    c = summary['conversations']
    assert (c['saved'], c['failed'], c['stopped_by_max_total_rounds']) == (2, 2, 2) and summary['exit_code'] == 0
    summary_file(tmp_path / 'on', summary, 'free')

    # the conversations are byte for byte those of a run without the notification; no line was added
    with_note = transcripts(files)
    monkeypatch.undo()
    summary_off, files_off = hm_like(tmp_path / 'off', monkeypatch, reply, notify=False)
    assert with_note == transcripts(files_off) and len(with_note) == 2
    assert summary_off['calls'] == summary['calls']
    for text in with_note.values():
        lines = text.decode('utf-8').splitlines()
        assert all(l.startswith(('Bot: ', 'User: ', 'System: <FAILED>')) for l in lines)


def test_imdb_like_normal_run_has_no_warning(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        summary, files = run_toy(tmp_path, monkeypatch, 'free')
    assert not warnings(caplog)
    assert all(summary['empty_after_filters'][r] == {'conversations': 0, 'empty_rounds': 0} for r in REASONS)
    assert summary['dispatch_stopped'] is None


def test_notification_outside_a_conversation_does_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        llm_calls.note_empty_filter('year_range_without_column', (1997, 2001), 'release_year', 'toy')
    assert not warnings(caplog)


def http_403():
    resp = requests.Response()
    resp.status_code = 403
    return requests.HTTPError('403 Client Error: Forbidden for url: https://example.invalid/infer', response=resp)


THREE = [('u1', 'tt5'), ('u2', 'tt11'), ('u3', 'tt17')]


def test_theta_403_stops_dispatching(tmp_path, monkeypatch, inject, caplog):
    """403 at a call without fallback (the opening request): that conversation is aborted as before, the next two
    are not started; the summary says why. Exit code and error counts as before."""
    inject({'extract_item_from_title': http_403()})
    with caplog.at_level(logging.ERROR):
        summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=THREE, extra={'openai_provider': 'thetaedgecloud'})
    c = summary['conversations']
    assert (c['dispatched'], c['aborted'], c['skipped'], c['saved']) == (3, 1, 2, 0)
    assert summary['skip_reasons'] == {llm_calls.THETA_403_STOP: 2}
    assert summary['dispatch_stopped'] == {'reason': llm_calls.THETA_403_STOP, 'provider': 'thetaedgecloud',
                                           'after_task': 'user u1 item tt5'}
    assert summary['llm_errors']['by_site'] == {'extract_item_from_title': 1} and summary['exit_code'] == 3
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and 'HTTP 403' in r.getMessage()]
    assert len(errors) == 1
    assert all(s in errors[0] for s in ('provider thetaedgecloud', '10-minute window', 'daily quota',
                                        'scripts/run_rate_limited.py', '--num-workers', 'rerun the unfinished'))
    summary_file(tmp_path, summary, 'free')


def test_theta_403_stops_later_strategies_too(tmp_path, monkeypatch, inject):
    """Two strategies: the 403 in the first conversation of user_item_pairs stops the whole run; the rest of that
    strategy and all of item_list are skipped with the same reason (item_list would run without the 403)."""
    def rule(site):
        rec = llm_calls.current()
        if site == 'extract_item_from_title' and rec is not None and (rec.user_id, rec.item_id) == ('u1', 'tt5'):
            return http_403()
        return None
    inject(rule)
    pd.DataFrame({'item_id': ['tt7', 'tt8', 'tt9']}).to_csv(tmp_path / 'items_list.csv', index=False)
    summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=THREE,
                             extra={'input_items_file': str(tmp_path / 'items_list.csv'),
                                    'openai_provider': 'thetaedgecloud'})
    assert summary['strategies'] == ['user_item_pairs', 'item_list']
    ps = summary['per_strategy']
    assert (ps['user_item_pairs']['dispatched'], ps['user_item_pairs']['aborted'], ps['user_item_pairs']['skipped']) == (3, 1, 2)
    assert (ps['item_list']['dispatched'], ps['item_list']['skipped']) == (3, 3) and 'saved' not in ps['item_list']
    assert summary['skip_reasons'] == {llm_calls.THETA_403_STOP: 5}
    assert summary['dispatch_stopped'] == {'reason': llm_calls.THETA_403_STOP, 'provider': 'thetaedgecloud',
                                           'after_task': 'user u1 item tt5'}
    assert not finals(files) and summary['exit_code'] == 3


def test_theta_403_at_a_fallback_site_is_unchanged(tmp_path, monkeypatch, inject):
    """A 403 swallowed by a fallback does not reach the task runner: the conversations go on with fallback text."""
    inject({'rephrase_phrase': http_403()})
    summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=THREE)
    c = summary['conversations']
    assert (c['saved'], c['skipped']) == (3, 0) and summary['dispatch_stopped'] is None
    assert summary['fallbacks']['by_site']['rephrase_phrase'] >= 3 and summary['exit_code'] == 3


def test_other_http_errors_do_not_stop_dispatching(tmp_path, monkeypatch, inject):
    resp = requests.Response()
    resp.status_code = 500
    inject({'extract_item_from_title': requests.HTTPError('500 Server Error', response=resp)})
    summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=THREE)
    c = summary['conversations']
    assert (c['aborted'], c['skipped']) == (3, 0) and summary['dispatch_stopped'] is None


PER_MINUTE = 'Rate limit reached for gpt-4o-mini on requests per min (RPM): Limit 500, Used 500, Requested 1.'
PER_DAY = 'Rate limit reached for gpt-4o-mini on requests per day (RPD): Limit 10000, Used 10000, Requested 1.'
NO_QUOTA = 'You exceeded your current quota, please check your plan and billing details.'


def rate_limit_error(message=PER_MINUTE, code=None):
    import openai
    try:
        import httpx2 as httpx        # the HTTP library of openai 3.x
    except ImportError:
        import httpx                  # openai 1.x / 2.x
    resp = httpx.Response(429, request=httpx.Request('POST', 'https://example.invalid/v1/chat/completions'))
    body = {'message': message, 'type': 'requests', 'code': code} if code else None
    return openai.RateLimitError(message, response=resp, body=body)


def test_403_with_another_provider_does_not_stop(tmp_path, monkeypatch, inject):
    """The quota error is looked up per provider: a requests.HTTPError 403 means nothing for azure."""
    inject({'extract_item_from_title': http_403()})
    summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=THREE, extra={'openai_provider': 'azure'})
    c = summary['conversations']
    assert (c['aborted'], c['skipped']) == (3, 0) and summary['dispatch_stopped'] is None


@pytest.mark.parametrize('error, reason, words', [
    (lambda: rate_limit_error(PER_DAY), llm_calls.OPENAI_DAILY_STOP,
     ('Daily rate limit', 'provider azure', 'next day', 'tomorrow', 'another API key')),
    (lambda: rate_limit_error(NO_QUOTA, code='insufficient_quota'), llm_calls.OPENAI_QUOTA_STOP,
     ('insufficient_quota', 'provider azure', 'does not come back by itself', 'quota / billing')),
], ids=['per_day', 'insufficient_quota'])
def test_azure_quota_errors_stop_the_whole_run(tmp_path, monkeypatch, inject, caplog, error, reason, words):
    inject({'extract_item_from_title': error()})
    pd.DataFrame({'item_id': ['tt7', 'tt8']}).to_csv(tmp_path / 'items_list.csv', index=False)
    with caplog.at_level(logging.ERROR):
        summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=THREE,
                                 extra={'openai_provider': 'azure', 'input_items_file': str(tmp_path / 'items_list.csv')})
    ps = summary['per_strategy']
    assert (ps['user_item_pairs']['aborted'], ps['user_item_pairs']['skipped'], ps['item_list']['skipped']) == (1, 2, 2)
    assert summary['skip_reasons'] == {reason: 4}
    assert summary['dispatch_stopped'] == {'reason': reason, 'provider': 'azure', 'after_task': 'user u1 item tt5'}
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and 'provider azure' in r.getMessage()]
    assert len(errors) == 1 and all(w in errors[0] for w in words), errors
    assert summary['exit_code'] == 3


def test_azure_per_minute_rate_limit_drops_only_that_conversation(tmp_path, monkeypatch, inject, caplog):
    """As in v1: a per-minute RateLimitError (after the SDK's retries) drops the conversation with a WARNING; the next
    ones are started."""
    def rule(site):
        rec = llm_calls.current()
        if site == 'extract_item_from_title' and rec is not None and rec.item_id == 'tt5':
            return rate_limit_error(PER_MINUTE)
        return None
    inject(rule)
    with caplog.at_level(logging.WARNING):
        summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=THREE, extra={'openai_provider': 'azure'})
    c = summary['conversations']
    assert (c['aborted'], c['skipped'], c['saved']) == (1, 0, 2) and summary['dispatch_stopped'] is None
    assert any(r.levelno == logging.WARNING and 'Error processing user u1 item tt5' in r.getMessage()
               for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno == logging.ERROR and 'provider azure' in r.getMessage()]


def test_rate_limit_error_with_theta_is_an_ordinary_error(tmp_path, monkeypatch, inject):
    """For thetaedgecloud only HTTP 403 is registered: an openai.RateLimitError at a chat call does not stop the run,
    not even a per-day one."""
    inject({'extract_item_from_title': rate_limit_error(PER_DAY)})
    summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=THREE, extra={'openai_provider': 'thetaedgecloud'})
    c = summary['conversations']
    assert (c['aborted'], c['skipped']) == (3, 0) and summary['dispatch_stopped'] is None


def test_quota_lookup_table():
    assert llm_calls.is_quota_exhausted(http_403(), 'thetaedgecloud') == llm_calls.THETA_403_STOP
    assert llm_calls.is_quota_exhausted(llm_calls.LLMCallFailed('x', http_403()), 'ThetaEdgeCloud') == llm_calls.THETA_403_STOP
    for provider in ('openai', 'azure', 'azure_foundry'):
        assert llm_calls.is_quota_exhausted(rate_limit_error(PER_DAY), provider) == llm_calls.OPENAI_DAILY_STOP
        assert llm_calls.is_quota_exhausted(rate_limit_error(NO_QUOTA, 'insufficient_quota'),
                                            provider) == llm_calls.OPENAI_QUOTA_STOP
        assert llm_calls.is_quota_exhausted(rate_limit_error(PER_MINUTE), provider) is None
        assert llm_calls.is_quota_exhausted(http_403(), provider) is None
    assert llm_calls.is_quota_exhausted(rate_limit_error(PER_DAY), 'thetaedgecloud') is None
    assert llm_calls.is_quota_exhausted(http_403(), 'sentence-transformers') is None
    # the embeddings provider answers for embedding call sites, the chat provider for the others
    config = {'openai_provider': 'thetaedgecloud', 'embeddings_provider': 'azure'}
    emb, chat = rate_limit_error(), rate_limit_error()
    emb.omnisim_llm_site, chat.omnisim_llm_site = 'query_embedding', 'generate_chit_chat'
    assert llm_calls.provider_of(emb, config) == 'azure' and llm_calls.provider_of(chat, config) == 'thetaedgecloud'


def test_fail_mode_with_403_is_unchanged(tmp_path, monkeypatch, inject):
    """on_llm_error: fail already stops after the first error; its skip reason stays."""
    inject({'rephrase_phrase': http_403()})
    summary, files = run_toy(tmp_path, monkeypatch, 'free', pairs=THREE, extra={'on_llm_error': 'fail'})
    assert summary['skip_reasons'] == {'on_llm_error=fail': 2} and summary['dispatch_stopped'] is None
