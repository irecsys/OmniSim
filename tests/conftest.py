import pytest

import utils.retrieval.local_index as li


@pytest.fixture(autouse=True)
def _no_real_embedding_client(monkeypatch):
    """The index builder can call the paid embedding API, and this machine may have a .env
    with real keys. No test may reach the real client: tests pass a fake embed_fn instead."""
    def refuse(config):
        pytest.fail('a test reached the real embedding API client (local_index._api_embed_fn); pass embed_fn')
    monkeypatch.setattr(li, '_api_embed_fn', refuse)
