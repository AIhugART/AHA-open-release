"""Start the gateway under uvicorn. Refuses to start in a way the profile does not allow (config.validate_for_serve)."""

from __future__ import annotations

import logging

import uvicorn

from . import runtime
from .app import create_app
from .config import Config, validate_for_serve
from .model_install import InstallManager


def make_server(cfg: Config, log_level: str = "warning", installer: InstallManager | None = None) -> uvicorn.Server:
    validate_for_serve(cfg)
    kwargs: dict = {}
    if cfg.profile == "lan":
        kwargs.update(ssl_certfile=str(cfg.tls_cert_path), ssl_keyfile=str(cfg.tls_key_path))
    config = uvicorn.Config(
        create_app(cfg, installer), host=cfg.listen_host or "127.0.0.1", port=cfg.listen_port, log_level=log_level,
        server_header=False, date_header=False, **kwargs,
    )
    return uvicorn.Server(config)


def serve(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = make_server(cfg, "info")
    handler = runtime.install_gateway_log(cfg)
    runtime.write_run_record(cfg)
    try:
        server.run()
    finally:
        runtime.remove_run_record(cfg)
        runtime.remove_gateway_log(handler)
