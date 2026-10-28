"""
utils.llm_calls — the single entry point for chat / embedding calls of the dialogue run.

Every call site calls ``chat(client, site, **kwargs)`` / ``embed(client, site, **kwargs)`` instead of
``client.chat.completions.create(**kwargs)`` / ``client.embeddings.create(**kwargs)``. The kwargs, the return
value and the exceptions are exactly those of the client; the wrapper only counts, per call site, the calls and
the errors (into the record of the current conversation, or of the run when no conversation is active).
An error is marked with ``exc.omnisim_llm_site`` and re-raised unchanged.

The places that replace a failed call by fallback content call ``fallback(site, exc, value, config)``: one
WARNING (secrets masked), one count, and the fallback value is returned.

Conversations: ``simulate_free`` / ``simulate_adaptive`` are wrapped by ``@conversation``, which starts a fresh
record in the worker thread, and in ``finally`` merges it exactly once into the active ``RunSummary``. A
conversation stopped by an LLM / embedding error keeps its partial transcript as ``<tmp name>.aborted`` with a
``System: <ABORTED> ...`` line. A saved conversation with fallbacks ends with a ``System: <FALLBACK> ...`` line.

Run: ``run_simulation`` creates a ``RunSummary`` (``activate_run``); it is written to
``{outputs_folder}/{dataset}/{mode}/run_summary_{run_ts}.json`` and gives run.py its exit code (0, or 3 after any
LLM / embedding error).

tests/test_llm_calls_static.py keeps the dialogue modules from calling the clients directly.
"""
import datetime
import functools
import inspect
import json
import logging
import os
import re
import threading
from collections import Counter

logger = logging.getLogger(__name__)

# call site -> what it is; the names are the keys of the per-site counts
SITES = {
    # chat calls with fallback content (the 10 LLM fallbacks)
    'expand_query_for_search':            'utils.expand_query_for_search',
    'generate_chit_chat':                 'utils.generate_chit_chat',
    'generate_rejection_explanation':     'utils.generate_rejection_explanation',
    'generate_recommendation_explanation': 'utils.generate_recommendation_explanation',
    'generate_bot_closing_failed':        'utils.generate_bot_closing_failed',
    'rephrase_phrase':                    'utils.rephrase_phrase',
    'free_user_short_reply':              'utils.generate_user_responses_free_mode, short reply',
    'free_user_details_reply':            'utils.generate_user_responses_free_mode, detailed reply',
    'attribute_question':                 'simulator.simulate_adaptive, attribute question (static / adaptive)',
    'profile_demographic':                'user_profile_builder, demographic summary',
    'profile_likes_long':                 'user_profile_builder, long-term preference summary',
    'profile_likes_short':                'user_profile_builder, short-term preference summary',
    # chat calls without fallback (an error stops the conversation)
    'extract_item_from_title':            'utils.extract_item_from_title (opening request)',
    'extract_dynamic_description':        'utils.extract_dynamic_description_from_chat',
    'extract_candidate_attributes':       'utils.extract_attributes_for_asking, candidates (adaptive)',
    'filter_attributes':                  'utils.extract_attributes_for_asking, filter (adaptive)',
    'extract_attribute_value':            'utils.extract_attribute_value',
    'rephrase_attribute':                 'utils.rephrase_attribute_to_natural_language',
    'startup_check':                      'quick_start.run_simulation, connection test',
    # embedding calls (no fallback)
    'profile_embedding':                  'simulator._get_user_embeddings',
    'query_embedding':                    'simulator, query embedding before each retrieval',
}

_SECRET_ENV = ('AZURE_KEY', 'OPENAI_KEY', 'THETA_KEY', 'DEEPSEEK_KEY', 'GEMINI_KEY')
_SECRET_PATTERNS = [
    re.compile(r'(?i)\bbearer\s+[^\s\'",;]+'),
    re.compile(r'(?i)\b(api[-_]?key|access[-_]?token|token|authorization|key)\b(\s*[=:]\s*)[^\s\'",;&]+'),
    re.compile(r'\bsk-[A-Za-z0-9_\-]{6,}'),
]
MAX_MESSAGE = 300


def sanitize(message) -> str:
    """The text of an exception on one line, with keys masked (mask_secrets) and cut to MAX_MESSAGE characters."""
    text = mask_secrets(' '.join(str(message).split()))
    return text if len(text) <= MAX_MESSAGE else text[:MAX_MESSAGE - 3] + '...'


def mask_secrets(text) -> str:
    """Masks 'Bearer …', 'api-key=…' / 'key: …' style pairs, 'sk-…' tokens, and the values of the key environment
    variables (read from os.environ only to compare; never logged or written). Keeps line breaks (tracebacks)."""
    text = str(text)
    for name in _SECRET_ENV:
        value = os.environ.get(name)
        if value and len(value) >= 6:
            text = text.replace(value, '***')
    text = _SECRET_PATTERNS[0].sub('Bearer ***', text)
    text = _SECRET_PATTERNS[1].sub(lambda m: f'{m.group(1)}{m.group(2)}***', text)
    text = _SECRET_PATTERNS[2].sub('sk-***', text)
    return text


class RunAborted(Exception):
    """Stops run_simulation before any conversation is dispatched (stage: 'startup_check' | 'profiles')."""

    def __init__(self, stage, error):
        super().__init__(f'{stage}: {type(error).__name__}: {sanitize(error)}')
        self.stage = stage
        self.error = error


ON_LLM_ERROR = ('fallback', 'fail')


class LLMCallFailed(Exception):
    """on_llm_error: fail — raised by fallback() instead of returning fallback text; stops the conversation."""

    def __init__(self, site, original):
        super().__init__(f'{site}: {type(original).__name__}: {sanitize(original)}')
        self.site = site
        self.original = original
        self.omnisim_llm_site = site      # an LLM error for is_llm_error / the conversation scope


def check_on_llm_error(config):
    value = str(config.get('on_llm_error') or 'fallback').strip().lower()
    if value not in ON_LLM_ERROR:
        raise ValueError(f"Config error: 'on_llm_error' must be one of {', '.join(ON_LLM_ERROR)}, "
                         f"got {config.get('on_llm_error')!r}.")
    return value


def _error_type(exc):
    """The type name reported for an error (the original one for LLMCallFailed)."""
    return type(exc.original).__name__ if isinstance(exc, LLMCallFailed) else type(exc).__name__


class Record:
    """Calls, errors and fallbacks of one conversation (or of the run, outside any conversation)."""

    def __init__(self, strategy=None, user_id=None, item_id=None):
        self.lock = threading.Lock()
        self.strategy, self.user_id, self.item_id = strategy, user_id, item_id
        self.calls = Counter()            # site -> logical calls (one chat() / embed(), including failed ones)
        self.errors = Counter()           # (site, error type) -> errors
        self.retries = Counter()          # (site, HTTP status) -> client-side retries inside one logical call
        self.fallbacks = Counter()        # site -> fallback values used
        self.fallback_log = []            # [(site, error type)] in order
        self.first_error = None           # (site, error type, sanitized message)
        self.tmp_file = None
        self.final_file = None
        self.succeed = None
        self.stopped_by_round_limit = False
        self.target_not_found = False     # target item not in df_items (note_target_not_found)
        self.empty_filters = {}           # reason -> {'value', 'column', 'rounds'} (note_empty_filter)
        self.status = None                # saved | aborted | dropped
        self.abort_file = None

    def error(self, site, exc):
        with self.lock:
            self.errors[(site, type(exc).__name__)] += 1
            if self.first_error is None:
                self.first_error = (site, type(exc).__name__, sanitize(exc))


_local = threading.local()
_dummy = Record()                         # calls outside any conversation and any run (scripts, tests)
_run = [None]                             # the active RunSummary


def current():
    """The record of the conversation running in this thread, or None."""
    return getattr(_local, 'record', None)


def _target():
    rec = current()
    if rec is not None:
        return rec
    return _run[0].outside if _run[0] is not None else _dummy


def _invoke(site, fn, kwargs):
    """The actual client call (one place, so offline tests can inject an error at a given site)."""
    return fn(**kwargs)


def _call(site, fn, kwargs):
    if site not in SITES:
        raise ValueError(f'unregistered LLM call site {site!r} (utils.llm_calls.SITES)')
    rec = _target()
    with rec.lock:
        rec.calls[site] += 1
    previous = getattr(_local, 'site', None)
    _local.site = site
    try:
        return _invoke(site, fn, kwargs)
    except Exception as e:
        try:
            e.omnisim_llm_site = site
        except Exception:
            pass
        rec.error(site, e)
        raise
    finally:
        _local.site = previous


def chat(client, site, **kwargs):
    """client.chat.completions.create(**kwargs), counted under `site`."""
    return _call(site, client.chat.completions.create, kwargs)


def embed(client, site, **kwargs):
    """client.embeddings.create(**kwargs), counted under `site`."""
    return _call(site, client.embeddings.create, kwargs)


def note_retry(status):
    """Called by a client that retries an HTTP request inside one logical call (the ThetaEdgeCloud client
    retries 409 / 429 / 503). Counted separately from the logical calls."""
    rec = _target()
    with rec.lock:
        rec.retries[(getattr(_local, 'site', None) or 'unknown', int(status))] += 1


def fallback(site, exc, value, config=None):
    """Called in the except clause of a call that has fallback content: one WARNING, one count, `value` back.
    With on_llm_error: fail, raises LLMCallFailed instead (the conversation stops)."""
    if config is not None and str(config.get('on_llm_error') or 'fallback').strip().lower() == 'fail':
        raise LLMCallFailed(site, exc) from exc
    rec = _target()
    logger.warning(f"LLM call failed, using fallback text: site={site} error={type(exc).__name__} "
                   f"message={sanitize(exc)}")
    with rec.lock:
        rec.fallbacks[site] += 1
        rec.fallback_log.append((site, type(exc).__name__))
    return value


def fallback_log():
    """[(site, error type)] of the fallbacks recorded so far by the current conversation (or the run)."""
    return _target().fallback_log


def startup_failed(exc):
    """The connection test at the start of run_simulation failed: the run stops (whatever on_llm_error is)."""
    raise RunAborted('startup_check', exc)


def is_llm_error(exc):
    return hasattr(exc, 'omnisim_llm_site')


# ── per-conversation scope ────────────────────────────────────────────────────

def attach_tmp(tmp_file):
    rec = current()
    if rec is not None:
        rec.tmp_file = tmp_file


def note_round_limit():
    rec = current()
    if rec is not None:
        rec.stopped_by_round_limit = True


def note_target_not_found():
    rec = current()
    if rec is not None:
        rec.target_not_found = True


# hard filters that leave no candidate for a structural reason: reason -> (constraint, why)
EMPTY_FILTER_REASONS = {
    'year_range_without_column': ('year_range', "the dataset has no '{column}' column, so every item is filtered out"),
    'require_any_genre_no_match': ('require_any_genre', "no item's '{column}' contains these genre words"),
}


def note_empty_filter(reason, value, column, dataset=None):
    """Called by the retrieval when a hard filter on its own left no candidate for a reason that repeats on every
    retrieval of the conversation (utils.retrieval.filters.build_mask `reasons`). Counts the empty retrieval; logs one
    WARNING per conversation and reason. Outside a conversation it does nothing. Visibility only: the retrieval
    result, the dialogue and the calls are unchanged."""
    rec = current()
    if rec is None:
        return
    with rec.lock:
        entry = rec.empty_filters.get(reason)
        first = entry is None
        if first:
            entry = rec.empty_filters[reason] = {'value': value, 'column': column, 'rounds': 0}
        entry['rounds'] += 1
    if first:
        constraint, why = EMPTY_FILTER_REASONS[reason]
        logger.warning(f"Hard filter left no candidates: {constraint}={value!r} — {why.format(column=column)} "
                       f"(dataset {dataset}, user {rec.user_id}, item {rec.item_id}). Every retrieval of this "
                       f"conversation with this constraint returns nothing; logged once per conversation.")


# ── quota / rate-limit errors that stop the whole run ─────────────────────────

THETA_403_STOP = 'ThetaEdgeCloud HTTP 403 (quota assumed used up)'
OPENAI_DAILY_STOP = 'OpenAI/Azure daily rate limit (RateLimitError, per day)'
OPENAI_QUOTA_STOP = 'OpenAI/Azure insufficient_quota (account quota used up)'
EMBEDDING_SITES = ('profile_embedding', 'query_embedding')


def _openai_daily_limit(exc):
    """A RateLimitError for a per-day limit (the condition of v1's daily rate-limit branch). Per-minute limits
    (requests / tokens per minute, still failing after the SDK's retries) are not registered: the conversation is
    dropped and the run goes on, as in v1."""
    import openai
    text = str(exc).lower()
    return isinstance(exc, openai.RateLimitError) and ('per day' in text or 'rpd' in text)


def _openai_insufficient_quota(exc):
    """A RateLimitError with code insufficient_quota: the account's quota is used up and does not come back by itself."""
    import openai
    return isinstance(exc, openai.RateLimitError) and getattr(exc, 'code', None) == 'insufficient_quota'


def _http_403(exc):
    import requests
    if not isinstance(exc, requests.HTTPError):
        return False
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status is None:                         # an HTTPError built without a response: its text starts with the status
        status = 403 if str(exc).startswith('403 ') else None
    return status == 403


# provider -> ((test on the error, stop reason), ...), first match wins. When a provider is added, register here the
# quota errors that should stop the whole run.
_OPENAI_QUOTA = ((_openai_insufficient_quota, OPENAI_QUOTA_STOP), (_openai_daily_limit, OPENAI_DAILY_STOP))
QUOTA_ERRORS = {
    'openai':         _OPENAI_QUOTA,
    'azure':          _OPENAI_QUOTA,
    'azure_foundry':  _OPENAI_QUOTA,
    'thetaedgecloud': ((_http_403, THETA_403_STOP),),  # Theta answers 403 when its on-demand quota is used up
}


def is_quota_exhausted(exc, provider):
    """The stop reason when `exc` is a registered quota error of `provider` (QUOTA_ERRORS), else None.
    LLMCallFailed (on_llm_error: fail) is unwrapped first."""
    exc = exc.original if isinstance(exc, LLMCallFailed) else exc
    for test, reason in QUOTA_ERRORS.get(str(provider or '').strip().lower(), ()):
        if test(exc):
            return reason
    return None


def provider_of(exc, config):
    """The provider that made the failed call: the embeddings provider for an embedding call site, else the chat
    provider (both read from the config as utils.get_openai_clients does)."""
    from utils.utils import resolve_embeddings_provider
    if getattr(exc, 'omnisim_llm_site', None) in EMBEDDING_SITES:
        return resolve_embeddings_provider(config)
    return str(config.get('openai_provider', 'openai')).lower()


def fallback_line(rec=None):
    """'<FALLBACK> fallbacks=N sites=a:1,b:2 errors=T:3' for the current conversation, or None without fallbacks.
    No brackets or quotes: tools scan conversation lines for '(item id)' and quoted titles."""
    rec = rec or current()
    if rec is None or not rec.fallbacks:
        return None
    errors = Counter(t for _, t in rec.fallback_log)
    return (f"<FALLBACK> fallbacks={sum(rec.fallbacks.values())} "
            f"sites={','.join(f'{k}:{v}' for k, v in sorted(rec.fallbacks.items()))} "
            f"errors={','.join(f'{k}:{v}' for k, v in sorted(errors.items()))}")


def saved(final_file, succeed):
    rec = current()
    if rec is not None:
        rec.final_file, rec.succeed = final_file, int(succeed)


def _abort_transcript(rec, exc):
    """Keep the partial transcript of a conversation stopped by an LLM / embedding error as <tmp>.aborted."""
    if not rec.tmp_file or not os.path.exists(rec.tmp_file):
        return None
    site = getattr(exc, 'omnisim_llm_site', None) or getattr(exc, 'site', None) or 'unknown'
    mode = _run[0].on_llm_error if _run[0] is not None else 'fallback'
    with open(rec.tmp_file, 'a', encoding='utf-8') as f:
        f.write(f"System: <ABORTED> site={site} error={_error_type(exc)} on_llm_error={mode}\n")
    target = os.path.splitext(rec.tmp_file)[0] + '.aborted'
    os.replace(rec.tmp_file, target)
    return target


def conversation(func):
    """Decorator of simulate_free / simulate_adaptive: a fresh record for this conversation, merged exactly once
    (in finally) into the active RunSummary. An LLM / embedding error that stops the conversation keeps the
    transcript as .aborted; any other exception leaves the tmp file as before."""
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if current() is not None:          # already inside a conversation (not expected; keep one record)
            return func(*args, **kwargs)
        a = sig.bind_partial(*args, **kwargs).arguments
        config = a.get('config') or {}
        rec = Record(config.get('generation_strategy'), a.get('user_id'), a.get('target_item_id'))
        _local.record = rec
        try:
            result = func(*args, **kwargs)
            rec.status = 'saved' if rec.final_file else 'dropped'
            return result
        except BaseException as e:
            if is_llm_error(e):
                rec.status = 'aborted'
                try:
                    rec.abort_file = _abort_transcript(rec, e)
                except OSError as oe:
                    logger.warning(f"could not keep the aborted transcript {rec.tmp_file}: {oe}")
            else:
                rec.status = 'dropped'
            if rec.first_error is None:
                rec.first_error = (getattr(e, 'omnisim_llm_site', None), _error_type(e), sanitize(e))
            try:
                e.omnisim_merged = True
            except Exception:
                pass
            raise
        finally:
            _local.record = None
            if _run[0] is not None:
                _run[0].add_conversation(rec)
    wrapper.omnisim_conversation_scope = True
    return wrapper


# ── per-run summary ───────────────────────────────────────────────────────────

def activate_run(summary):
    _run[0] = summary


def active_run():
    return _run[0]


def deactivate_run():
    _run[0] = None


def _merge_counter(into, other):
    for k, v in other.items():
        into[k] += v


class RunSummary:
    """Counts of one run_simulation call; merged from the conversations under a lock, written by the main thread."""

    def __init__(self, config, config_files=None):
        from utils.utils import resolve_embeddings_provider
        self.lock = threading.Lock()
        self.config = config
        self.config_files = list(config_files or [])
        self.started = datetime.datetime.now()
        self.on_llm_error = str(config.get('on_llm_error') or 'fallback').strip().lower()
        self.stop = threading.Event()     # on_llm_error: fail — set after the first stopped conversation
        self.providers = {'chat_provider': str(config.get('openai_provider', 'openai')).lower(),
                          'chat_model': config.get('chat_model'),
                          'embeddings_provider': resolve_embeddings_provider(config),
                          'embeddings_model': config.get('embeddings_model')}
        self.outside = Record()
        self.strategies = []
        self.status, self.aborted_at, self.crash = 'completed', None, None
        self.startup = {'ok': None}
        self.profiles = {'rebuilt': False, 'cache_written': None, 'failed_users': []}
        self.counts = Counter()           # dispatched saved succeeded failed stopped_by_max_total_rounds target_not_found
                                          # aborted dropped skipped
        self.per_strategy = {}
        self.calls, self.errors, self.retries, self.fallbacks = Counter(), Counter(), Counter(), Counter()
        self.fallback_errors = Counter()
        self.conversations_with_fallback = 0
        self.affected = []
        self.skip_reasons = Counter()
        self.empty_filters = {r: Counter() for r in EMPTY_FILTER_REASONS}   # reason -> conversations, empty_rounds
        self.dispatch_stop = None         # why no further task was started (a quota error, is_quota_exhausted)
        self.path = None

    # main thread
    def dispatched(self, strategy, n):
        with self.lock:
            if strategy not in self.strategies:
                self.strategies.append(strategy)
            self.counts['dispatched'] += n
            self.per_strategy.setdefault(strategy, Counter())['dispatched'] += n

    def skipped(self, strategy, reason):
        with self.lock:
            self.counts['skipped'] += 1
            self.skip_reasons[reason] += 1
            self.per_strategy.setdefault(strategy, Counter())['skipped'] += 1

    def dropped_outside(self, strategy, identifier, exc):
        """An exception in a task before its conversation record started (or outside simulate_*)."""
        with self.lock:
            self.counts['dropped'] += 1
            self.per_strategy.setdefault(strategy, Counter())['dropped'] += 1
            self.affected.append({'strategy': strategy, 'task': identifier, 'status': 'dropped', 'file': None,
                                  'error': type(exc).__name__, 'message': sanitize(exc)})

    def _rel(self, path):
        if not path:
            return None
        base = self.config.get('outputs_folder') or ''
        try:
            return os.path.relpath(path, base) if base else path
        except ValueError:
            return path

    # worker threads
    def add_conversation(self, rec):
        with self.lock:
            st = self.per_strategy.setdefault(rec.strategy, Counter())
            self.counts[rec.status] += 1
            st[rec.status] += 1
            if rec.status == 'saved':
                self.counts['succeeded' if rec.succeed else 'failed'] += 1
            if rec.stopped_by_round_limit:
                self.counts['stopped_by_max_total_rounds'] += 1
            if rec.target_not_found:
                self.counts['target_not_found'] += 1
            for reason, entry in rec.empty_filters.items():
                self.empty_filters[reason]['conversations'] += 1
                self.empty_filters[reason]['empty_rounds'] += entry['rounds']
            _merge_counter(self.calls, rec.calls)
            _merge_counter(self.errors, rec.errors)
            _merge_counter(self.retries, rec.retries)
            _merge_counter(self.fallbacks, rec.fallbacks)
            _merge_counter(self.fallback_errors, Counter(t for _, t in rec.fallback_log))
            if rec.fallbacks:
                self.conversations_with_fallback += 1
                st['fallbacks'] += sum(rec.fallbacks.values())
            if rec.fallbacks or rec.status != 'saved':
                entry = {'strategy': rec.strategy, 'user_id': None if rec.user_id is None else str(rec.user_id),
                         'item_id': None if rec.item_id is None else str(rec.item_id), 'status': rec.status,
                         'file': self._rel(rec.final_file or rec.abort_file or rec.tmp_file),
                         'fallbacks': dict(sorted(rec.fallbacks.items())),
                         'errors': {f'{s}:{t}': n for (s, t), n in sorted(rec.errors.items())}}
                if rec.first_error:
                    entry['first_error'] = {'site': rec.first_error[0], 'error': rec.first_error[1],
                                            'message': rec.first_error[2]}
                self.affected.append(entry)

    def stop_dispatch(self, reason, task, provider=None):
        """Record why no new task may start for the rest of the run (a quota error, is_quota_exhausted): _run_parallel
        then skips the remaining tasks of this strategy and every later strategy. The first reason is kept."""
        with self.lock:
            if self.dispatch_stop is None:
                self.dispatch_stop = {'reason': reason, 'provider': provider, 'after_task': task}

    # profiles, startup (main thread)
    def profile_result(self, rebuilt, failures, cache_written):
        with self.lock:
            self.profiles = {'rebuilt': bool(rebuilt), 'cache_written': cache_written,
                             'failed_users': [{'user_id': str(u), 'site': s, 'error': t} for u, s, t in failures]}

    def aborted(self, stage, exc):
        self.status, self.aborted_at = 'aborted', stage
        if stage == 'startup_check':
            self.startup = {'ok': False, 'error': _error_type(exc), 'message': sanitize(exc)}
        elif stage == 'profiles':
            self.profiles = {'rebuilt': True, 'cache_written': False, 'failed_users': [],
                             'stopped_at': {'site': getattr(exc, 'site', None), 'error': _error_type(exc),
                                            'message': sanitize(exc)}}

    def crashed(self, exc):
        self.status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'crashed'
        self.crash = {'error': type(exc).__name__, 'message': sanitize(exc)}

    # result
    def llm_error_total(self):
        errors = Counter(self.errors)
        _merge_counter(errors, self.outside.errors)
        return sum(errors.values())

    def exit_code(self):
        """3 after any LLM / embedding error (fallback, aborted or dropped conversation, profile failure,
        failed startup check, skipped tasks); 0 otherwise. Conversations that end as failed are normal."""
        bad = (sum(self.fallbacks.values()) + sum(self.outside.fallbacks.values()) + self.counts['aborted']
               + self.counts['dropped'] + self.counts['skipped'] + len(self.profiles['failed_users'])
               + (self.startup.get('ok') is False) + (self.status == 'aborted'))
        if self.status in ('crashed', 'interrupted'):
            return 1
        return 3 if bad else 0

    def as_dict(self):
        def by_site(counter):
            out = Counter()
            for (site, _), n in counter.items():
                out[site] += n
            return dict(sorted(out.items()))

        def by_type(counter):
            out = Counter()
            for (_, t), n in counter.items():
                out[t] += n
            return dict(sorted(out.items()))
        calls, errors, retries, fallbacks = Counter(self.calls), Counter(self.errors), Counter(self.retries), \
            Counter(self.fallbacks)
        _merge_counter(calls, self.outside.calls)
        _merge_counter(errors, self.outside.errors)
        _merge_counter(retries, self.outside.retries)
        _merge_counter(fallbacks, self.outside.fallbacks)
        fb_errors = Counter(self.fallback_errors)
        _merge_counter(fb_errors, Counter(t for _, t in self.outside.fallback_log))
        c = self.counts
        return {
            'library_version': self.config.get('library_version'),
            'run_timestamp': self.config.get('run_timestamp') or self.started.strftime('%Y%m%d%H%M%S'),
            'started': self.started.isoformat(timespec='seconds'),
            'finished': datetime.datetime.now().isoformat(timespec='seconds'),
            'dataset': self.config.get('dataset'), 'mode': self.config.get('mode_refinement'),
            'strategies': list(self.strategies), 'config_files': self.config_files,
            **self.providers, 'on_llm_error': self.on_llm_error,
            'status': self.status, 'aborted_at': self.aborted_at, 'crash': self.crash,
            'exit_code': self.exit_code(),
            'conversations': {k: c[k] for k in ('dispatched', 'saved', 'succeeded', 'failed',
                                                'stopped_by_max_total_rounds', 'target_not_found',
                                                'aborted', 'dropped', 'skipped')},
            'skip_reasons': dict(self.skip_reasons),
            'dispatch_stopped': self.dispatch_stop,
            'empty_after_filters': {
                **{r: {'conversations': c['conversations'], 'empty_rounds': c['empty_rounds']}
                   for r, c in self.empty_filters.items()},
                'note': 'retrievals left with no candidate by a hard filter for a structural reason; '
                        'one WARNING per conversation and reason; the dialogue itself is unchanged'},
            'calls': {'total': sum(calls.values()), 'by_site': dict(sorted(calls.items())),
                      'note': 'logical calls: one llm_calls.chat / embed each, failed ones included'},
            'retries': {'total': sum(retries.values()), 'by_site': by_site(retries),
                        'by_status': {str(k): v for k, v in sorted(by_type(retries).items())},
                        'note': 'HTTP retries inside one logical call by the ThetaEdgeCloud client (409 / 429 / 503); '
                                'the OpenAI / Azure SDK retries internally (max_retries) without being counted'},
            'llm_errors': {'total': sum(errors.values()), 'by_site': by_site(errors), 'by_error_type': by_type(errors),
                           'outside_conversations': {f'{s}:{t}': n for (s, t), n in sorted(self.outside.errors.items())}},
            'fallbacks': {'conversations_with_fallback': self.conversations_with_fallback,
                          'total': sum(fallbacks.values()), 'by_site': dict(sorted(fallbacks.items())),
                          'by_error_type': dict(sorted(fb_errors.items()))},
            'startup_check': self.startup,
            'profiles': self.profiles,
            'affected': self.affected,
            'per_strategy': {k: dict(v) for k, v in self.per_strategy.items()},
        }

    def summary_path(self):
        cfg = self.config
        run_ts = cfg.get('run_timestamp') or self.started.strftime('%Y%m%d%H%M%S')
        mode = str(cfg.get('mode_refinement') or 'unknown').strip().lower()
        return os.path.join(cfg.get('outputs_folder') or 'chats', str(cfg.get('dataset')), mode,
                            f'run_summary_{run_ts}.json')

    def write(self):
        data = self.as_dict()
        self.path = self.summary_path()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            f.write('\n')
        c, fb = data['conversations'], data['fallbacks']
        line = (f"Run summary: status={data['status']} exit_code={data['exit_code']} saved={c['saved']} "
                f"(succeeded {c['succeeded']}, failed {c['failed']}) aborted={c['aborted']} dropped={c['dropped']} "
                f"skipped={c['skipped']} fallbacks={fb['total']} in {fb['conversations_with_fallback']} conversation(s) "
                f"llm_errors={data['llm_errors']['total']} profile_failures={len(data['profiles']['failed_users'])}")
        (logger.warning if data['exit_code'] else logger.info)(line)
        logger.info(f"Run summary written to {self.path}")
        return data
