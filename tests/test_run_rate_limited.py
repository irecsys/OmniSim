"""
Unit tests for scripts/run_rate_limited.py: Theta rate limit, call recording, failure
classification. No network: requests is replaced by fakes.
"""
import json
import subprocess
import sys

import pytest

from scripts import run_rate_limited as lim


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def time(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class FakeResp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {'body': {'infer_requests': [{'output': {'message': 'Rephrased!'}}]}}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error: Forbidden for url: x")

    def json(self):
        return self._body


class FakeRequests:
    def __init__(self, statuses=(), exc=None):
        self.statuses = list(statuses)
        self.exc = exc
        self.urls = []

    def post(self, url, *a, **kw):
        self.urls.append(url)
        if self.exc:
            raise self.exc
        return FakeResp(self.statuses.pop(0) if self.statuses else 200)


THETA = 'https://ondemand.thetaedgecloud.com/infer_request/qwen3_8_flash/completions'


def test_window_limits_and_is_shared_between_instances(tmp_path):
    clock = FakeClock()
    path = str(tmp_path / 'w.json')
    a = lim.ThetaWindow(path, '3/10', clock.time, clock.sleep)
    b = lim.ThetaWindow(path, '3/10', clock.time, clock.sleep)     # another process, same file
    a.gate(); b.gate(); a.gate()
    assert clock.slept == []
    b.gate()                                                        # 4th request in the window waits
    assert clock.slept == [pytest.approx(10.5)] and b.waited == pytest.approx(10.5)
    data = json.load(open(path))
    assert data['total'] == 4 and len(data['ts']) == 1              # the first three have expired


def test_prefilled_window_blocks_until_it_expires(tmp_path):
    clock = FakeClock()
    path = str(tmp_path / 'w.json')
    json.dump({'ts': [clock.t - 100] * 95, 'total': 0}, open(path, 'w'))
    lim.ThetaWindow(path, '95/600', clock.time, clock.sleep).gate()
    assert clock.slept == [pytest.approx(600 - 100 + 0.5)]


def test_init_window_prefill_and_no_overwrite(tmp_path, capsys):
    path = str(tmp_path / 'w.json')
    lim.init_window(path, ['105@2026-10-23T23:20:12', '1@now'])
    d = json.load(open(path))
    assert len(d['ts']) == 106 and d['prefill'] == ['105@2026-10-23T23:20:12', '1@now']
    lim.init_window(path, ['5@now'])
    assert len(json.load(open(path))['ts']) == 106 and 'not prefilled again' in capsys.readouterr().out


def test_shim_gates_and_records_theta_only(tmp_path):
    rec = lim.Recorder(str(tmp_path / 'r.json'))
    win = lim.ThetaWindow(str(tmp_path / 'w.json'))
    real = FakeRequests([429, 200, 403])
    shim = lim.ThetaRequestsShim(real, win, rec)
    for _ in range(3):
        shim.post(THETA)
    shim.post('https://example.invalid/embeddings')            # not Theta: not counted
    d = json.load(open(tmp_path / 'r.json'))
    assert d['theta_http'] == 3 and d['theta_status'] == {'429': 1, '200': 1, '403': 1}
    assert json.load(open(tmp_path / 'w.json'))['total'] == 3


def test_shim_records_network_exception_and_reraises(tmp_path):
    rec = lim.Recorder(str(tmp_path / 'r.json'))
    shim = lim.ThetaRequestsShim(FakeRequests(exc=TimeoutError('read timed out')), lim.ThetaWindow(str(tmp_path / 'w.json')), rec)
    with pytest.raises(TimeoutError):
        shim.post(THETA)
    assert 'timed out' in json.load(open(tmp_path / 'r.json'))['theta_exceptions'][0]


def test_swallowed_403_is_recorded_and_classified_infra(tmp_path, monkeypatch):
    """A past incident: a Theta 403 inside rephrase_phrase is swallowed by v1 (fallback = the phrase
    itself), the dialogue goes on — the recorder still sees the failed call."""
    import utils.utils as uu
    rec = lim.Recorder(str(tmp_path / 'r.json'))
    monkeypatch.setattr(uu, '_requests', lim.ThetaRequestsShim(FakeRequests([403]), lim.ThetaWindow(str(tmp_path / 'w.json')), rec))
    client = lim.ChatClientProxy(uu.ThetaEdgeCloudClient('key', 'qwen3_8_flash'), rec)
    out = uu.rephrase_phrase({'chat_model': 'm'}, client, 'Ready to find something great?')
    assert out == 'Ready to find something great?'                   # v1 fallback, no exception visible
    d = json.load(open(tmp_path / 'r.json'))
    assert d['chat_calls'] == 1 and len(d['chat_errors']) == 1 and '403' in d['chat_errors'][0]['error']
    assert d['theta_status'] == {'403': 1}
    assert lim.classify(d) == 'infra_failed'


def test_successful_call_and_retried_429_are_ok(tmp_path, monkeypatch):
    import utils.utils as uu
    rec = lim.Recorder(str(tmp_path / 'r.json'))
    monkeypatch.setattr(uu, '_requests', lim.ThetaRequestsShim(FakeRequests([429, 200]), lim.ThetaWindow(str(tmp_path / 'w.json')), rec))
    monkeypatch.setattr('time.sleep', lambda s: None)                # the client's back-off
    client = lim.ChatClientProxy(uu.ThetaEdgeCloudClient('key', 'qwen3_8_flash'), rec)
    assert uu.rephrase_phrase({'chat_model': 'm'}, client, 'Hello') == 'Rephrased!'
    d = json.load(open(tmp_path / 'r.json'))
    assert d['theta_http'] == 2 and d['chat_calls'] == 1 and d['chat_errors'] == []
    assert lim.classify(d) == 'ok'


def test_embedding_errors_are_infra(tmp_path):
    rec = lim.Recorder(str(tmp_path / 'r.json'))

    class Emb:
        class embeddings:
            @staticmethod
            def create(**kw):
                raise RuntimeError('azure 500')
    client = lim.EmbeddingsClientProxy(Emb(), rec)
    with pytest.raises(RuntimeError):
        client.embeddings.create(input='x', model='m')
    d = json.load(open(tmp_path / 'r.json'))
    assert d['embedding_calls'] == 1 and lim.classify(d) == 'infra_failed'


def test_install_wraps_clients_in_process_only(tmp_path, monkeypatch):
    import utils.quick_start as qs
    import utils.user_profile_builder as upb
    import utils.utils as uu
    for mod in (qs, upb):                                            # restored after the test
        monkeypatch.setattr(mod, 'get_openai_clients', mod.get_openai_clients)
    monkeypatch.setattr(uu, '_requests', uu._requests)

    class C:
        chat = type('c', (), {'completions': object()})()
        embeddings = object()
    monkeypatch.setattr(uu, 'get_openai_clients', lambda config: (C(), C()))
    rec = lim.Recorder(str(tmp_path / 'r.json'))
    lim.install(rec, lim.ThetaWindow(str(tmp_path / 'w.json')))
    chat, emb = qs.get_openai_clients({})
    assert isinstance(chat, lim.ChatClientProxy) and isinstance(emb, lim.EmbeddingsClientProxy)
    assert isinstance(upb.get_openai_clients({})[0], lim.ChatClientProxy)
    assert isinstance(uu._requests, lim.ThetaRequestsShim)


def test_cli_classify(tmp_path):
    for errors, want in (([], 'ok'), ([{'error': '403'}], 'infra_failed')):
        p = tmp_path / 'r.json'
        p.write_text(json.dumps({'chat_errors': errors, 'embedding_errors': []}))
        out = subprocess.run([sys.executable, 'scripts/run_rate_limited.py', '--classify', str(p)],
                             capture_output=True, text=True, check=True).stdout.strip()
        assert out == want


def test_wrapper_requires_a_record_path(monkeypatch):
    monkeypatch.delenv('OMNISIM_CALL_RECORD', raising=False)
    with pytest.raises(SystemExit):
        lim.main(['--config', 'x'])
