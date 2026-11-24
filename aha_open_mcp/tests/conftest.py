"""The harness is tested against a REAL gateway (`aha-open serve` child process, fake worker); nothing is mocked."""

from __future__ import annotations

import os
import tempfile

# Before aha_open_mcp is imported: its state folder must not be the one a person's sandbox uses.
os.environ.setdefault("AHA_OPEN_MCP_STATE_DIR", tempfile.mkdtemp(prefix="aha-open-mcp-test-state-"))

import pytest  # noqa: E402

from aha_open_mcp import sandbox  # noqa: E402


@pytest.fixture(scope="module")
def sandbox_up():
    info = sandbox.start()
    try:
        yield info
    finally:
        sandbox.stop()


@pytest.fixture
def stopped():
    """No sandbox running."""
    sandbox.stop()
    yield
    sandbox.stop()
