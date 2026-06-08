from __future__ import annotations

from pathlib import Path

import pytest

from computecop.config import (
    CONFIG_ENV_VAR,
    ConfigError,
    ConfigSource,
    load_config,
    load_effective_config,
    resolve_config_path,
)


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPUTECOP_ENDPOINTS", raising=False)
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    config = load_config()
    assert config.server.host == "127.0.0.1"
    assert config.policy.ram_yield_percent == 85.0
    assert config.endpoints


def test_load_config_rejects_invalid_endpoint_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPUTECOP_ENDPOINTS", "{bad-json")
    with pytest.raises(ConfigError):
        load_config()


def test_toml_config_overrides_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "computecop.toml"
    config_file.write_text(
        """
[server]
port = 9001

[policy]
ram_yield_percent = 80.0
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    effective = load_effective_config(config_path=config_file)
    assert effective.config.server.port == 9001
    assert effective.config.policy.ram_yield_percent == 80.0
    assert effective.sources["server.port"] == ConfigSource.TOML
    assert effective.sources["policy.ram_yield_percent"] == ConfigSource.TOML
    assert effective.sources["server.host"] == ConfigSource.DEFAULT


def test_environment_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "computecop.toml"
    config_file.write_text("[server]\nport = 9001\n", encoding="utf-8")
    monkeypatch.setenv("COMPUTECOP_PORT", "9100")
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    effective = load_effective_config(config_path=config_file)
    assert effective.config.server.port == 9100
    assert effective.sources["server.port"] == ConfigSource.ENVIRONMENT


def test_cli_overrides_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "computecop.toml"
    config_file.write_text("[server]\nport = 9001\n", encoding="utf-8")
    monkeypatch.setenv("COMPUTECOP_PORT", "9100")
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    effective = load_effective_config(
        config_path=config_file,
        cli_overrides={"server": {"port": 9200}},
    )
    assert effective.config.server.port == 9200
    assert effective.sources["server.port"] == ConfigSource.CLI


def test_computecop_config_env_resolves_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "computecop.toml"
    config_file.write_text("[server]\nhost = \"10.0.0.5\"\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))
    effective = load_effective_config()
    assert effective.config.server.host == "10.0.0.5"
    assert effective.config_path == config_file
    assert effective.sources["server.host"] == ConfigSource.TOML


def test_resolve_config_path_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(ConfigError, match="config file not found"):
        resolve_config_path(missing)


def test_resolve_config_path_rejects_missing_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.toml"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(missing))
    with pytest.raises(ConfigError, match="points to missing file"):
        resolve_config_path()


def test_toml_endpoints_replace_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "computecop.toml"
    config_file.write_text(
        """
[[endpoints]]
name = "custom"
kind = "ollama"
base_url = "http://127.0.0.1:11435"
health_path = "/api/tags"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    config = load_config(config_path=config_file)
    assert len(config.endpoints) == 1
    assert config.endpoints[0].name == "custom"


def test_invalid_toml_raises_config_error(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text("server = [", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(config_path=config_file)


def test_effective_config_explain_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "computecop.toml"
    config_file.write_text("[server]\nport = 9001\n", encoding="utf-8")
    monkeypatch.setenv("COMPUTECOP_HOST", "10.0.0.2")
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    effective = load_effective_config(config_path=config_file)
    entries = {entry["path"]: entry for entry in effective.explain_entries()}
    assert entries["server.port"]["source"] == ConfigSource.TOML.value
    assert entries["server.host"]["source"] == ConfigSource.ENVIRONMENT.value
    document = effective.explain_document()
    assert document["config_path"] == str(config_file)
    assert any(entry["path"] == "server.port" for entry in document["entries"])