import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(autouse=True)
def _no_literature_network(monkeypatch):
    """Tests never hit OpenAlex. Tests that need papers patch research.literature themselves."""
    for k in ("TIINGO_API_KEY", "FRED_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    import research.literature as L
    from research.providers import ProviderError

    def offline(url, **kw):
        raise ProviderError("offline in tests")
    monkeypatch.setattr(L, "_get", offline)
