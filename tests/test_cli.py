import pytest
import cli

class DummyApi:
    def __init__(self, *args, **kwargs):
        pass

class DummyPortfolio:
    def __init__(self, *args, **kwargs):
        raise AssertionError("Portfolio should not be created when signing key is missing")

def test_download_portfolio_requires_key(monkeypatch):
    monkeypatch.setattr(cli, "TradeRepublicApi", DummyApi)
    monkeypatch.setattr(cli, "Portfolio", DummyPortfolio)
    with pytest.raises(RuntimeError):
        cli.download_portfolio("123", "456")
