"""Operator configuration (config.toml) and the startup rules each network profile must satisfy."""

from __future__ import annotations

import ipaddress
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROFILES = ("local", "lan", "internet")
WILDCARD_HOSTS = {"", "0.0.0.0", "::", "[::]", "*"}
LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1", "[::1]"}


class ConfigError(Exception):
    """The configuration is not allowed to start the gateway (fails closed, with a sentence the operator can act on)."""


@dataclass
class ModelConfig:
    id: str
    worker: str
    variant: str = ""
    gpu: int = 0
    enabled: bool = True
    python: str = ""  # override of the worker's interpreter
    weights_dir: str = ""  # override of where the weights are


@dataclass
class Config:
    data_dir: Path
    profile: str = "local"
    listen: str = "127.0.0.1:8890"
    hosts: list[str] = field(default_factory=list)  # extra Host names the server answers to
    allowed_origins: list[str] = field(default_factory=list)
    require_token: bool = True
    enroll: bool = False
    access_confirmed: bool = False  # internet: the operator states Cloudflare Access sits in front
    hosted_service_obligations: str = ""  # "accepted" unlocks models that carry hosted-service duties
    tls_cert: str = ""
    tls_key: str = ""
    part_size: int = 50 * 1024 * 1024
    max_upload_bytes: int = 1024 * 1024 * 1024
    max_queue_per_client: int = 4
    max_queue_total: int = 32
    max_jobs_per_minute: int = 30
    result_ttl_s: int = 3600
    upload_ttl_s: int = 3600
    max_wait_for_swap_s: float = 60.0
    worker_load_timeout_s: float = 900.0
    job_timeout_s: float = 3600.0
    cancel_grace_s: float = 15.0
    models: list[ModelConfig] = field(default_factory=list)

    # ---- derived ----
    @property
    def listen_host(self) -> str:
        return split_listen(self.listen)[0]

    @property
    def listen_port(self) -> int:
        return split_listen(self.listen)[1]

    @property
    def tls_cert_path(self) -> Path:
        return Path(self.tls_cert) if self.tls_cert else self.data_dir / "tls" / "server.crt"

    @property
    def tls_key_path(self) -> Path:
        return Path(self.tls_key) if self.tls_key else self.data_dir / "tls" / "server.key"

    @property
    def local_token_path(self) -> Path:
        return self.data_dir / "local_client.token"

    def model(self, model_id: str) -> ModelConfig | None:
        return next((m for m in self.models if m.id == model_id), None)


def default_data_dir() -> Path:
    env = os.environ.get("AHA_OPEN_DATA_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AHA-open"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "aha-open"


def split_listen(listen: str) -> tuple[str, int]:
    listen = listen.strip()
    if listen.startswith("["):
        host, _, rest = listen[1:].partition("]")
        port = rest.lstrip(":")
        return host, int(port) if port else 8890
    host, _, port = listen.rpartition(":")
    if not host and not port:
        return "", 8890
    if not host:  # only a port was given
        return "", int(port)
    return host, int(port)


def is_loopback(host: str) -> bool:
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def from_dict(data: dict, data_dir: Path) -> Config:
    cfg = Config(data_dir=data_dir)
    top_scalar = {
        "profile", "listen", "hosts", "allowed_origins", "hosted_service_obligations", "tls_cert", "tls_key",
        "access_confirmed", "part_size", "max_upload_bytes", "max_queue_per_client", "max_queue_total",
        "max_jobs_per_minute", "result_ttl_s", "upload_ttl_s", "max_wait_for_swap_s", "worker_load_timeout_s",
        "job_timeout_s", "cancel_grace_s",
    }
    for key in top_scalar:
        if key in data:
            setattr(cfg, key, data[key])
    auth = data.get("auth", {})
    cfg.require_token = bool(auth.get("require_token", cfg.require_token))
    cfg.enroll = bool(auth.get("enroll", cfg.enroll))
    for raw in data.get("models", []):
        cfg.models.append(
            ModelConfig(
                id=raw["id"],
                worker=raw["worker"],
                variant=raw.get("variant", ""),
                gpu=int(raw.get("gpu", 0)),
                enabled=bool(raw.get("enabled", True)),
                python=raw.get("python", ""),
                weights_dir=raw.get("weights_dir", ""),
            )
        )
    if cfg.profile not in PROFILES:
        raise ConfigError(f"profile must be one of {PROFILES}, got {cfg.profile!r}")
    return cfg


def load(data_dir: Path | None = None) -> Config:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    path = data_dir / "config.toml"
    if not path.exists():
        raise ConfigError(f"{path} does not exist - run `aha-open init` first")
    with path.open("rb") as handle:
        return from_dict(tomllib.load(handle), data_dir)


def render(cfg: Config) -> str:
    """Write a config back as TOML (only the keys we own; enough for init and for tests)."""

    def q(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return "[" + ", ".join(q(v) for v in value) + "]"
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [
        f"profile = {q(cfg.profile)}",
        f"listen = {q(cfg.listen)}",
        f"hosts = {q(cfg.hosts)}",
        f"allowed_origins = {q(cfg.allowed_origins)}",
        f"access_confirmed = {q(cfg.access_confirmed)}",
        f"hosted_service_obligations = {q(cfg.hosted_service_obligations)}",
        f"max_queue_per_client = {cfg.max_queue_per_client}",
        f"max_queue_total = {cfg.max_queue_total}",
        f"result_ttl_s = {cfg.result_ttl_s}",
        "",
        "[auth]",
        f"require_token = {q(cfg.require_token)}",
        f"enroll = {q(cfg.enroll)}",
    ]
    for model in cfg.models:
        lines += [
            "",
            "[[models]]",
            f"id = {q(model.id)}",
            f"worker = {q(model.worker)}",
            f"variant = {q(model.variant)}",
            f"gpu = {model.gpu}",
            f"enabled = {q(model.enabled)}",
        ]
        if model.python:
            lines.append(f"python = {q(model.python)}")
        if model.weights_dir:
            lines.append(f"weights_dir = {q(model.weights_dir)}")
    return "\n".join(lines) + "\n"


def validate_for_serve(cfg: Config) -> None:
    """Refuse to start in a way that would expose the service more than the profile promises (docs/07_security.md section 3)."""
    host = cfg.listen_host
    if cfg.profile == "local":
        if not is_loopback(host):
            raise ConfigError("profile 'local' must listen on a loopback address (127.0.0.1)")
    elif cfg.profile == "lan":
        if host in WILDCARD_HOSTS:
            raise ConfigError("profile 'lan' must name the interface to listen on; 0.0.0.0 and :: are not allowed")
        if not cfg.tls_cert_path.exists() or not cfg.tls_key_path.exists():
            raise ConfigError("profile 'lan' requires HTTPS: run `aha-open tls init` to create a certificate")
        if not cfg.require_token:
            raise ConfigError("profile 'lan' requires a token for every client")
    elif cfg.profile == "internet":
        if not is_loopback(host):
            raise ConfigError("profile 'internet' must listen on loopback and be reached through a tunnel")
        if not cfg.access_confirmed:
            raise ConfigError(
                "profile 'internet' needs Cloudflare Access in front of the tunnel; set access_confirmed = true "
                "once it is configured (see docs/06_deployment.md)"
            )
        if not cfg.hosts:
            raise ConfigError("profile 'internet' needs `hosts` to list the public hostname the tunnel serves")
        if not cfg.require_token:
            raise ConfigError("profile 'internet' requires a token for every client")
    if cfg.profile != "local" and cfg.enroll and cfg.profile == "internet":
        # Allowed, but only because the operator asked for it explicitly (default is off).
        pass
