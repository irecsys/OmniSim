"""Optional passive call log of the chat client (llm_log_file): one JSONL line per returned chat request with the
call site, the request as passed, the reply text, usage and the conversation's user / item. It never changes the
request, the reply, the order of calls, the retries or the random state; write errors are counted, not raised."""
import json
import logging
import os
import threading
import time

from utils import llm_calls

logger = logging.getLogger(__name__)
_LOCKS = {}
_LOCKS_LOCK = threading.Lock()
REQUEST_KEYS = ('model', 'messages', 'temperature', 'max_tokens', 'top_p', 'seed', 'extra_body')


# One lock per log file, shared by all threads.
def _lock(path):
    with _LOCKS_LOCK:
        return _LOCKS.setdefault(os.path.abspath(path), threading.Lock())


# Token usage of a response as a plain dict (None when absent).
def _usage(resp):
    u = getattr(resp, 'usage', None)
    if u is None:
        return None
    if hasattr(u, 'model_dump'):
        return u.model_dump()
    return {k: getattr(u, k, None) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}


# Appends one JSONL line per returned chat request to the log file.
class CallLog:
    def __init__(self, path, mode=None):
        self.path = os.path.abspath(path)
        self.mode = mode
        self.lock = _lock(self.path)
        self.lines = 0
        self.errors = 0
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    # Writes the line for one request and its response; never raises.
    def record(self, kwargs, resp):
        try:
            rec = llm_calls.current()
            user = getattr(rec, 'user_id', None)
            item = getattr(rec, 'item_id', None)
            choices = []
            for ch in getattr(resp, 'choices', None) or []:
                msg = getattr(ch, 'message', None)
                choices.append({'message': {'role': 'assistant', 'content': getattr(msg, 'content', None)},
                                'finish_reason': getattr(ch, 'finish_reason', None)})
            line = {'ts': round(time.time(), 6), 'thread': threading.get_ident(),
                    'site': getattr(llm_calls._local, 'site', None),
                    'scope': f'user {user} item {item} mode {self.mode}#0' if rec is not None else 'run',
                    'user_id': None if user is None else str(user), 'item_id': None if item is None else str(item),
                    'mode': self.mode, 'model': kwargs.get('model'),
                    'request': {k: kwargs[k] for k in REQUEST_KEYS if k in kwargs},
                    'response': {'choices': choices, 'usage': _usage(resp)}}
            text = json.dumps(line, ensure_ascii=False, default=str)
            with self.lock:
                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(text + '\n')
                self.lines += 1
        except Exception as e:                    # never affects the call
            self.errors += 1
            if self.errors == 1:
                logger.warning(f'llm_log_file: a line could not be written ({type(e).__name__}: {e}); the call is not '
                               f'affected, later failures are only counted')


def from_config(config):
    """A CallLog for llm_log_file, or None when the key is absent or null."""
    path = config.get('llm_log_file')
    if path is None or str(path).strip().lower() in ('', 'none', 'null', '~'):
        return None
    return CallLog(str(path), config.get('mode_refinement'))
