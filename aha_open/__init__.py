"""AHA open: a gateway that serves open image/video models through the neutral /v1 API.

The gateway never imports torch; model code runs in separate worker processes (docs/04_architecture.md).
"""

__version__ = "0.1.0.dev1"
API_VERSION = "1.0"
