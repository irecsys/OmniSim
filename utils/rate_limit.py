"""Rate limits of openai_compatible chat calls: retry 429 / 5xx / timeouts / connection errors unchanged with
exponential backoff (RetryPolicy), and a request gate shared by all processes through a locked file (SharedWindow).
Keys and defaults: DEFAULTS below and docs/API.md; the jitter never touches the global `random` state."""
import email.utils
import fcntl
import json
import logging
import os
import random
import threading
import time

logger = logging.getLogger(__name__)

DEFAULTS = {'chat_retry_base': 2.0, 'chat_retry_max_wait': 300.0, 'chat_retry_total': 1800.0,
            'chat_rate_limit': '70/60', 'chat_rate_window_file': 'logs/openai_compatible_rate_window.json'}


def _retryable(exc):
    """The HTTP status to count (429, 5xx; 0 for a timeout or connection error) when `exc` is retried, else None."""
    import openai
    if isinstance(exc, openai.RateLimitError):
        return 429
    if isinstance(exc, openai.InternalServerError):
        return int(getattr(exc, 'status_code', 500) or 500)
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return 0
    return None


def _retry_after(exc):
    """Seconds from the response's Retry-After header (a number or an HTTP date), or None."""
    response = getattr(exc, 'response', None)
    headers = getattr(response, 'headers', None) or {}
    value = headers.get('retry-after') if hasattr(headers, 'get') else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        when = email.utils.parsedate_to_datetime(str(value))
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


# Exponential backoff for retryable errors (429, 5xx, timeouts, connection errors).
class RetryPolicy:
    def __init__(self, base=2.0, max_wait=300.0, total=1800.0, sleep=time.sleep, rng=None, note=None):
        self.base, self.max_wait, self.total = float(base), float(max_wait), float(total)
        self.sleep = sleep
        self.rng = rng or random.Random()          # its own generator: the global `random` state is never used
        self._rng_lock = threading.Lock()
        self.note = note

    def wait_for(self, attempt, exc):
        """The wait before retry number `attempt` (0-based): Retry-After when given, else base * 2**attempt with
        +-10% jitter; never more than max_wait."""
        after = _retry_after(exc)
        if after is not None:
            return min(self.max_wait, after)
        with self._rng_lock:
            jitter = self.rng.uniform(0.9, 1.1)
        return min(self.max_wait, self.base * (2 ** attempt) * jitter)

    def call(self, fn):
        """fn() with retries; the last error is raised when the next wait would exceed the total."""
        waited, attempt = 0.0, 0
        while True:
            try:
                return fn()
            except Exception as exc:
                status = _retryable(exc)
                if status is None:
                    raise
                wait = self.wait_for(attempt, exc)
                if waited + wait > self.total:
                    logger.error(f"openai_compatible: giving up after {attempt} retries and {waited:.0f} s of waiting "
                                 f"(HTTP {status or 'timeout / connection error'})")
                    raise
                attempt += 1
                waited += wait
                if self.note:
                    self.note(status)
                logger.warning(f"openai_compatible: rate-limit retry {attempt} after HTTP "
                               f"{status or 'timeout / connection error'}, waiting {wait:.1f} s "
                               f"({waited:.0f} s of {self.total:.0f} s)")
                self.sleep(wait)


def parse_rate(value):
    """(max_req, window seconds) of "N/S", or None for no gate ("0", "none", null, "")."""
    if value is None or str(value).strip().lower() in ('', '0', 'none', 'null', '~', 'off'):
        return None
    n, _, s = str(value).partition('/')
    n, s = int(n), float(s or 60)
    if n < 1 or s <= 0:
        raise ValueError(f"Config error: 'chat_rate_limit' must be N/S with N >= 1 and S > 0, got {value!r}.")
    return n, s


class SharedWindow:
    """At most max_req acquire() calls in any `window` seconds, shared by every process (and thread) that uses
    `path`: the timestamps of the last window live in a JSON file, read and written under an flock."""

    def __init__(self, path, max_req, window, clock=time.time, sleep=time.sleep):
        self.path = os.path.abspath(path)
        self.max_req, self.window = int(max_req), float(window)
        self.clock, self.sleep = clock, sleep
        self.waited = 0.0
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)

    def _try(self):
        """Under the lock: take a slot now (returns 0) or return the seconds until the oldest one leaves."""
        with open(self.path + '.lock', 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                try:
                    with open(self.path, encoding='utf-8') as f:
                        stamps = json.load(f).get('stamps', [])
                except (OSError, ValueError):
                    stamps = []
                now = self.clock()
                stamps = [t for t in stamps if now - t < self.window]
                if len(stamps) < self.max_req:
                    stamps.append(now)
                    wait = 0.0
                else:
                    wait = max(0.01, min(stamps) + self.window - now)
                tmp = f'{self.path}.{os.getpid()}.{threading.get_ident()}.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump({'stamps': stamps}, f)
                os.replace(tmp, self.path)
                return wait
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    # Blocks until a request slot is free in the shared window, then takes it.
    def acquire(self):
        while True:
            wait = self._try()
            if wait == 0.0:
                return
            self.waited += wait
            self.sleep(wait)


def from_config(config, note=None):
    """(RetryPolicy, SharedWindow or None) for an openai_compatible client."""
    get = lambda k: DEFAULTS[k] if config.get(k) is None else config.get(k)   # noqa: E731
    policy = RetryPolicy(get('chat_retry_base'), get('chat_retry_max_wait'), get('chat_retry_total'), note=note)
    # a key that is present with null / "0" / "none" turns the gate off; an absent key gives the default
    rate = parse_rate(config.get('chat_rate_limit') if 'chat_rate_limit' in config else DEFAULTS['chat_rate_limit'])
    gate = SharedWindow(get('chat_rate_window_file'), *rate) if rate else None
    return policy, gate
