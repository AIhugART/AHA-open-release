"""`aha-open start` / `stop`: a REAL gateway is started in the background on a free port and stopped again.

These are what the Windows installer's Start-menu shortcuts run, so the failure modes matter: a second click must not start a
second gateway, a foreign program on the port must not be killed or reported as ours, and a start that fails must say why.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from aha_open import launcher
from aha_open.config import Config, render


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_cfg(tmp_path: Path, profile: str = "local") -> Config:
    cfg = Config(data_dir=tmp_path / "data", profile=profile, listen=f"127.0.0.1:{free_port()}")
    cfg.data_dir.mkdir(parents=True)
    (cfg.data_dir / "config.toml").write_text(render(cfg), encoding="utf-8")
    return cfg


def cli(cfg: Config, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "aha_open.cli", "--data-dir", str(cfg.data_dir), *args],
                          capture_output=True, text=True, timeout=120, check=False)


@pytest.fixture
def cfg(tmp_path):
    c = make_cfg(tmp_path)
    yield c
    pid = launcher.recorded_pid(c)  # never leave a gateway behind, whatever the test did
    if pid:
        launcher._kill_tree(pid)


def test_start_runs_the_gateway_in_the_background_once_and_stop_ends_it(cfg):
    first = cli(cfg, "start", "--no-open")
    assert first.returncode == 0, first.stdout + first.stderr
    assert launcher.health_ok(cfg)
    pid = launcher.recorded_pid(cfg)
    assert pid and launcher.pid_alive(pid)  # the pid the gateway wrote itself

    again = cli(cfg, "start", "--no-open")  # a second click on the shortcut
    assert again.returncode == 0 and "already running" in again.stdout
    assert launcher.recorded_pid(cfg) == pid  # the same gateway, not a second one

    stopped = cli(cfg, "stop")
    assert stopped.returncode == 0 and "stopped" in stopped.stdout, stopped.stdout + stopped.stderr
    assert not launcher.pid_alive(pid) and not launcher.health_ok(cfg)
    assert launcher.recorded_pid(cfg) is None

    nothing = cli(cfg, "stop")
    assert nothing.returncode == 0 and "not running" in nothing.stdout


def test_a_program_that_owns_the_port_is_reported_and_not_replaced(cfg):
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", cfg.listen_port))
        squatter.listen()
        started = cli(cfg, "start", "--no-open")
        assert started.returncode == 1 and "used by another program" in started.stdout
        assert str(cfg.listen_port) in started.stdout
    assert launcher.recorded_pid(cfg) is None  # nothing of ours was started


def test_stop_leaves_a_foreign_server_on_our_port_alone(cfg):
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    foreign = HTTPServer(("127.0.0.1", cfg.listen_port), Health)
    threading.Thread(target=foreign.serve_forever, daemon=True).start()
    try:
        stopped = cli(cfg, "stop")
        assert stopped.returncode == 1 and "not started by this installation" in stopped.stdout
        assert launcher.health_ok(cfg)  # still answering: we did not touch it
    finally:
        foreign.shutdown()


def test_start_is_for_the_local_profile_only(tmp_path):
    cfg = make_cfg(tmp_path, profile="internet")
    started = launcher.start(cfg)
    assert started[0] is False and "local profile" in started[1] and launcher.recorded_pid(cfg) is None


def test_a_start_that_fails_says_why_from_the_gateways_own_output(cfg, monkeypatch):
    real_popen = subprocess.Popen

    def crashing(argv, **kwargs):  # the gateway dies at once with a message, like a configuration error would
        return real_popen([sys.executable, "-c", "print('error: the thing is wrong'); raise SystemExit(3)"], **kwargs)

    monkeypatch.setattr(launcher.subprocess, "Popen", crashing)
    ok, text = launcher.start(cfg)
    assert ok is False and "did not start" in text and "the thing is wrong" in text


def test_a_stale_record_after_a_crash_is_cleaned_not_trusted(cfg):
    path = cfg.data_dir / "run" / "gateway.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"pid": 4294967}', encoding="utf-8")  # no such process
    assert launcher.recorded_pid(cfg) is None
    ok, text = launcher.stop(cfg)
    assert ok and "not running" in text and not path.exists()


def test_stop_waits_until_the_process_is_really_gone(cfg, monkeypatch):
    """On POSIX SIGTERM only asks; even `taskkill` can return before the process is gone. Stop must not report 'stopped' early."""
    assert cli(cfg, "start", "--no-open").returncode == 0
    pid = launcher.recorded_pid(cfg)
    kill_for_real = launcher._kill_tree
    monkeypatch.setattr(launcher, "_kill_tree", lambda p: threading.Timer(0.8, kill_for_real, args=(p,)).start())  # dies 0.8 s later
    ok, text = launcher.stop(cfg)
    assert ok, text  # stop waited for the death instead of looking once and giving up
    assert not launcher.pid_alive(pid)
