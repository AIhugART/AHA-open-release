"""The request's ordered ``inputs`` -> the reference objects of the diffusers ``ref2va`` workflow.

The gateway gives each input as ``{"role", "path"}`` in the order the client sent them, and the order is part of the request: it
fixes the ``<Picture i>`` / ``<Video k>`` / ``<Audio j>`` labels of the prompt and the audio/video clock, so a different order is a
different request. Only local paths ever arrive here (the gateway stores uploads); a URL is refused so that the worker never
reaches out to the network on a client's behalf.
"""

from __future__ import annotations

ROLE_KIND = {"reference": "image", "reference_video": "video", "reference_audio": "audio"}


def from_inputs(inputs: list[dict]) -> list:
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    loaders = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
    references = []
    for item in inputs:
        kind = ROLE_KIND.get(item.get("role"))
        if kind is None:
            raise ValueError(f"input role {item.get('role')!r} is not a reference role of this model")
        path = str(item.get("path") or "")
        if path.startswith(("http://", "https://")):
            raise ValueError("a reference must be a file the gateway stored, not a URL")
        references.append(loaders[kind].from_file(path))
    if references and all(r.kind == "audio" for r in references):
        raise ValueError("an audio reference needs at least one image or video reference beside it")
    return references
