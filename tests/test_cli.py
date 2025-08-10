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


def test_pair_device(monkeypatch):
    class DummyPairApi:
        def __init__(self, *args, **kwargs):
            self.initiated = False
            self.token = None

        def initiate_device_reset(self):
            self.initiated = True

        def complete_device_reset(self, token):
            self.token = token

    dummy = DummyPairApi()
    monkeypatch.setattr(cli, "TradeRepublicApi", lambda *a, **kw: dummy)
    monkeypatch.setattr("builtins.input", lambda prompt="": "123456")
    cli.pair_device("123", "456")
    assert dummy.initiated
    assert dummy.token == "123456"


def test_pair_device_handles_missing_process_id(monkeypatch):
    class FailingPairApi:
        def __init__(self, *args, **kwargs):
            pass

        def initiate_device_reset(self):
            raise KeyError("processId")

    monkeypatch.setattr(cli, "TradeRepublicApi", lambda *a, **kw: FailingPairApi())
    with pytest.raises(RuntimeError):
        cli.pair_device("123", "456")
