"""Start the gateway under uvicorn. Refuses to start in a way the profile does not allow (config.validate_for_serve)."""

from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .config import Config, validate_for_serve


def make_server(cfg: Config, log_level: str = "warning") -> uvicorn.Server:
    validate_for_serve(cfg)
    kwargs: dict = {}
    if cfg.profile == "lan":
        kwargs.update(ssl_certfile=str(cfg.tls_cert_path), ssl_keyfile=str(cfg.tls_key_path))
    config = uvicorn.Config(
        create_app(cfg), host=cfg.listen_host or "127.0.0.1", port=cfg.listen_port, log_level=log_level,
        server_header=False, date_header=False, **kwargs,
    )
    return uvicorn.Server(config)


def serve(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    make_server(cfg, "info").run()
