"""`aha-open check`: each failure layer gets its own sentence (server off != wrong token != Access in front != certificate)."""

from __future__ import annotations

import http.server
import socket
import threading

import pytest
from conftest import new_client_token, start_gateway

from aha_open.cli import main
from aha_open.config import Config, ModelConfig
from aha_open.tlsutil import init_tls


def _check(capsys, *argv) -> tuple[int, str]:
    code = main(["check", *argv])
    return code, capsys.readouterr().out


def test_healthy_server_with_a_good_token(gw, capsys):
    code, out = _check(capsys, gw.base, "--token", gw.tokens["local"])
    assert code == 0
    assert "server is up" in out and "token accepted (profile local)" in out and "model example-image: ready" in out


def test_without_a_token_it_only_checks_reachability(gw, capsys):
    code, out = _check(capsys, gw.base)
    assert code == 0 and "server is up" in out and "--token" in out


def test_a_wrong_token_is_not_reported_as_a_dead_server(gw, capsys):
    code, out = _check(capsys, gw.base, "--token", "aho_wrong")
    assert code == 1 and "server is up" in out and "does not accept this token" in out


def test_a_server_that_is_off_says_so(capsys):
    with socket.socket() as s:  # a port that was free a moment ago and has nothing on it
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    code, out = _check(capsys, f"http://127.0.0.1:{port}")
    assert code == 1 and "nothing is listening" in out and "aha-open serve" in out


def test_a_name_that_does_not_resolve_says_so(capsys):
    code, out = _check(capsys, "https://no-such-host.invalid")
    assert code == 1 and "does not resolve" in out


class _Stub(http.server.BaseHTTPRequestHandler):
    mode = "access"

    def do_GET(self):
        if self.mode == "access":  # what Cloudflare Access does to a client without a valid service token
            self.send_response(302)
            self.send_header("Location", "https://team.cloudflareaccess.com/login")
            self.send_header("CF-Ray", "abc123")
            self.end_headers()
        elif self.mode == "cf403":
            self.send_response(403)
            self.send_header("CF-Ray", "abc123")
            self.end_headers()
        else:  # some other web server entirely
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html>welcome</html>")

    def log_message(self, *args):
        pass


@pytest.fixture
def stub():
    servers = []

    def make(mode: str) -> str:
        handler = type("H", (_Stub,), {"mode": mode})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield make
    for server in servers:
        server.shutdown()


@pytest.mark.parametrize("mode", ["access", "cf403"])
def test_cloudflare_access_in_front_is_named(stub, capsys, mode):
    code, out = _check(capsys, stub(mode))
    assert code == 1 and "Cloudflare Access stopped the request" in out and "--cf-id" in out


def test_something_that_is_not_aha_open_is_not_called_healthy(stub, capsys):
    code, out = _check(capsys, stub("html"))
    assert code == 1 and "not with a /v1 health document" in out


def test_lan_certificate_must_be_pinned(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AHA_FAKE_LOAD_S", "0.1")
    cfg = Config(data_dir=tmp_path / "lan", profile="lan", listen="127.0.0.1:0", models=[ModelConfig("example-image", "fake")])
    cert = init_tls(cfg)
    running = start_gateway(cfg)
    try:
        code, out = _check(capsys, running.base)
        assert code == 1 and "certificate is not trusted" in out and "--cacert" in out
        token = new_client_token(running, "laptop")
        code, out = _check(capsys, running.base, "--cacert", str(cert), "--token", token)
        assert code == 0 and "token accepted (profile lan)" in out
    finally:
        running.stop()


def test_tunnel_config_names_the_hostname_and_port(gw, capsys):
    assert main(["--data-dir", str(gw.cfg.data_dir / "not-initialised"), "tunnel", "--hostname", "aha-open.example.com"]) == 0
    out = capsys.readouterr().out
    assert "hostname: aha-open.example.com" in out and "http://127.0.0.1:" in out and 'profile = "internet"' in out
    assert "TLS ends there" in out  # the privacy trade-off is stated where the operator makes the choice
