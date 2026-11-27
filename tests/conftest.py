"""Global pytest guard: tests must never hit the network.

If a test forgets to mock an HTTP call, this fixture makes it fail loudly
instead of silently leaking a real API key into a traceback.
"""
import httpx
import pytest


@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch):
    def _deny(*args, **kwargs):
        raise AssertionError(
            "outbound HTTP attempted during test -- mock it (see tests/conftest.py)"
        )

    # Block at the httpx module level. Per-test monkeypatch (e.g. http_fetch
    # tests) can still override httpx.get in that test scope.
    monkeypatch.setattr(httpx, "request", _deny)
    monkeypatch.setattr(httpx, "get", _deny)
    monkeypatch.setattr(httpx, "post", _deny)