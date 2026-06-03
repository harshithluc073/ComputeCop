from __future__ import annotations

import pytest

from computecop.config import ConfigError, load_config


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPUTECOP_ENDPOINTS", raising=False)
    config = load_config()
    assert config.server.host == "127.0.0.1"
    assert config.policy.ram_yield_percent == 85.0
    assert config.endpoints


def test_load_config_rejects_invalid_endpoint_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPUTECOP_ENDPOINTS", "{bad-json")
    with pytest.raises(ConfigError):
        load_config()
