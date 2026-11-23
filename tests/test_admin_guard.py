"""The one rule that decides who may use /admin. Pure function, so every combination can be checked."""

from __future__ import annotations

import pytest

from aha_open.app import admin_allowed


@pytest.mark.parametrize(
    ("profile", "peer", "host", "allowed"),
    [
        ("local", "127.0.0.1", "127.0.0.1:8890", True),
        ("local", "127.0.0.1", "localhost:8890", True),
        ("local", "::1", "[::1]:8890", True),
        ("local", "192.168.1.20", "127.0.0.1:8890", False),  # another machine on the network
        ("local", None, "127.0.0.1:8890", False),  # no peer address at all
        ("local", "127.0.0.1", "aha.example.com", False),  # a public name, e.g. one served through a tunnel
        ("local", "127.0.0.1", "", False),
        ("internet", "127.0.0.1", "127.0.0.1:8890", False),  # the tunnel's own path looks exactly like a local visit
        ("internet", "127.0.0.1", "aha.example.com", False),
        ("lan", "127.0.0.1", "127.0.0.1:8890", False),
        ("lan", "192.168.1.20", "192.168.1.5:8890", False),
    ],
)
def test_admin_allowed(profile, peer, host, allowed):
    assert admin_allowed(profile, peer, host) is allowed
