"""
run_rate_limited.py - run run.py with a ThetaEdgeCloud rate limit and record every LLM / embedding call.
The wrappers are installed in this process only; library code is not modified. Linux and macOS only (fcntl).

run.py has no rate limit of its own, and ThetaEdgeCloud's on-demand quota was 100 requests per 10 minutes
at the time of this release. Use this script for longer runs with openai_provider: thetaedgecloud and
--num-workers 1.

  rate limit   every POST to thetaedgecloud (retries included) passes a gate: at most MAX_REQ requests
               in any WINDOW seconds (OMNISIM_THETA_RATE, default 95/600). The timestamps live in a JSON
               file (OMNISIM_THETA_WINDOW, default logs/theta_window.json) under an flock, so consecutive
               run.py processes share one window.
  record       every chat / embedding call made through the clients from get_openai_clients (the
               startup "Hello" check included) and every Theta HTTP status are written to a JSON
               record (OMNISIM_CALL_RECORD, required). A call that raises is recorded even when the
               library replaces it with fallback text. classify(record) is 'infra_failed' when any call
               raised, otherwise 'ok'.

Usage:
    OMNISIM_CALL_RECORD=<record.json> python scripts/run_rate_limited.py <run.py arguments>
    python scripts/run_rate_limited.py --classify <record.json>      # prints ok | infra_failed
    python scripts/run_rate_limited.py --init-window <file> --prefill N@<ISO time> [--prefill M@now]
"""
import fcntl
import json
import os
import sys
import threading
import time
from datetime import datetime

DEFAULT_WINDOW_FILE = 'logs/theta_window.json'
DEFAULT_RATE = '95/600'
RETRIED_STATUSES = (409, 429, 503)          # utils.utils._ThetaCompletions retries these


class ThetaWindow:
    """At most max_req gate() calls in any `window` seconds, shared by processes through `path`."""

    def __init__(self, path, rate=DEFAULT_RATE, clock=time.time, sleep=time.sleep):
        self.path = path
        self.max_req, self.window = int(rate.split('/')[0]), float(rate.split('/')[1])
        self.clock, self.sleep = clock, sleep
        self.waited = 0.0
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def _locked(self, fn):
        with open(self.path + '.lock', 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                data = json.load(open(self.path, encoding='utf-8')) if os.path.exists(self.path) else {}
                result = fn(data)
                tmp = self.path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                os.replace(tmp, self.path)
                return result
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def gate(self):
        while True:
            def take(data):
                now = self.clock()
                ts = [t for t in data.get('ts', []) if now - t < self.window]
                if len(ts) < self.max_req:
                    ts.append(now)
                    data['ts'] = ts
                    data['total'] = data.get('total', 0) + 1
                    return 0.0
                data['ts'] = ts
                return self.window - (now - min(ts)) + 0.5
            wait = self._locked(take)
            if wait <= 0:
                return
            self.waited += wait
            self.sleep(wait)


class Recorder:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {'theta_http': 0, 'theta_status': {}, 'theta_exceptions': [],
                     'chat_calls': 0, 'chat_errors': [], 'embedding_calls': 0, 'embedding_errors': [],
                     'rate_limit_wait_s': 0.0, 'started': datetime.now().isoformat(timespec='seconds')}
        self.save()

    def update(self, fn):
        with self.lock:
            fn(self.data)
            self.save()

    def save(self):
        if self.path:
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=1)
            os.replace(tmp, self.path)


def classify(record):
    """'infra_failed' when any LLM / embedding call raised, else 'ok'."""
    return 'infra_failed' if record['chat_errors'] or record['embedding_errors'] else 'ok'


class ThetaRequestsShim:
    """Replaces the `requests` module inside utils.utils: Theta POSTs are gated and recorded."""

    def __init__(self, real, window, recorder):
        self._real, self._window, self._rec = real, window, recorder

    def post(self, url, *args, **kwargs):
        if 'thetaedgecloud' not in str(url):
            return self._real.post(url, *args, **kwargs)
        self._window.gate()
        def count(d):
            d['theta_http'] += 1
            d['rate_limit_wait_s'] = round(self._window.waited, 1)
        self._rec.update(count)
        try:
            resp = self._real.post(url, *args, **kwargs)
        except Exception as e:
            self._rec.update(lambda d: d['theta_exceptions'].append(repr(e)[:300]))
            raise
        code = str(resp.status_code)
        self._rec.update(lambda d: d['theta_status'].__setitem__(code, d['theta_status'].get(code, 0) + 1))
        return resp

    def __getattr__(self, name):
        return getattr(self._real, name)


class _CallProxy:
    def __init__(self, inner, recorder, kind):
        self._inner, self._rec, self._kind = inner, recorder, kind

    def create(self, *args, **kwargs):
        self._rec.update(lambda d: d.__setitem__(f'{self._kind}_calls', d[f'{self._kind}_calls'] + 1))
        try:
            return self._inner.create(*args, **kwargs)
        except Exception as e:
            self._rec.update(lambda d: d[f'{self._kind}_errors'].append(
                {'at': datetime.now().isoformat(timespec='seconds'), 'error': repr(e)[:300]}))
            raise


class ChatClientProxy:
    def __init__(self, client, recorder):
        self._client = client
        self.chat = type('chat', (), {})()
        self.chat.completions = _CallProxy(client.chat.completions, recorder, 'chat')

    def __getattr__(self, name):
        return getattr(self._client, name)


class EmbeddingsClientProxy:
    def __init__(self, client, recorder):
        self._client = client
        self.embeddings = _CallProxy(client.embeddings, recorder, 'embedding')

    def __getattr__(self, name):
        return getattr(self._client, name)


def install(recorder, window):
    """Install the shim and the client proxies into the already-imported library modules."""
    import utils.quick_start as qs
    import utils.user_profile_builder as upb
    import utils.utils as uu
    if not isinstance(uu._requests, ThetaRequestsShim):
        uu._requests = ThetaRequestsShim(uu._requests, window, recorder)
    orig = uu.get_openai_clients

    def clients(config):
        chat, emb = orig(config)
        return ChatClientProxy(chat, recorder), EmbeddingsClientProxy(emb, recorder)
    qs.get_openai_clients = clients
    upb.get_openai_clients = clients


def init_window(path, prefills):
    """Create the window file with prefilled timestamps ('N@<ISO time>' or 'N@now'); refuse if it exists."""
    if os.path.exists(path):
        print(f"{path} exists — not prefilled again")
        return
    ts = []
    for spec in prefills:
        n, when = spec.split('@')
        t = time.time() if when == 'now' else datetime.fromisoformat(when).timestamp()
        ts += [t] * int(n)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'ts': ts, 'total': 0, 'prefill': prefills}, f)
    print(f"{path}: prefilled {len(ts)} requests ({', '.join(prefills)})")


def main(argv):
    if argv[:1] == ['--classify']:
        print(classify(json.load(open(argv[1], encoding='utf-8'))))
        return 0
    if argv[:1] == ['--init-window']:
        init_window(argv[1], [argv[i + 1] for i, a in enumerate(argv) if a == '--prefill'])
        return 0
    record_path = os.environ.get('OMNISIM_CALL_RECORD')
    if not record_path:
        sys.exit('OMNISIM_CALL_RECORD must name the per-run record file')
    sys.path.insert(0, os.getcwd())
    recorder = Recorder(record_path)
    window = ThetaWindow(os.environ.get('OMNISIM_THETA_WINDOW', DEFAULT_WINDOW_FILE),
                         os.environ.get('OMNISIM_THETA_RATE', DEFAULT_RATE))
    install(recorder, window)
    import runpy
    sys.argv = ['run.py'] + argv
    runpy.run_path('run.py', run_name='__main__')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
