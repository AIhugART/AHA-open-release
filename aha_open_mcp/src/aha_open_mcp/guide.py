"""What an error code MEANS and what to DO about it (plan D10).

The table is a second copy of knowledge that lives in the gateway, so it is guarded in BOTH directions by tests: every code the
gateway can emit (read from its source, not copied) and every code the specification lists must have an entry here, and an
entry for a code nobody emits is a defect too. Different causes get different actions (the server is off != the token is wrong
!= the model is still loading)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import aha_open
from aha_open.scheduler import WORKER_ERROR_CODES

REPO = Path(__file__).resolve().parents[3]

ERROR_CODE_GUIDE: dict[str, tuple[str, str]] = {
    "invalid_request": ("The request itself is wrong: a missing field, or a parameter outside what the model's `capabilities` allow.",
                        "Fix the request; read the error message for the field. Retrying the same request will fail the same way."),
    "install_running": ("A model download is already running on this server (the web page's Download panel allows one at a time).",
                        "Wait for it to finish, or cancel it from the same panel, then ask again."),
    "already_installed": ("That model is already installed, so there is nothing to download.",
                          "Use it as it is (GET /v1/manifest lists it); reinstall only on purpose, from the command line."),
    "variant_not_installed": ("The variant to switch to is not (completely) on this server's disk.",
                              "Download it first (the web page's Models panel, or `aha-open models install <model> --variant <name>`)."),
    "model_busy": ("A job of this model is waiting or running, so its variant cannot be switched or deleted now.",
                   "Wait for the job to finish, or cancel it, then try again."),
    "unsupported_task":("The model does not do this task (for example video on an image model).",
                         "Pick a model whose `tasks` list contains it (GET /v1/manifest)."),
    "upload_incomplete": ("The upload was not completed, a part is missing, or the assembled file's sha256 differs from the one announced.",
                          "Upload again, send every part, then call complete; check the sha256 you announce."),
    "unauthorized": ("The token is missing or wrong. This is NOT 'the server is off' and NOT 'the model is loading'.",
                     "Use the token in the data directory's local_client.token (local profile) or pair again."),
    "forbidden": ("The token is valid but this is not allowed here (a blocked Origin or Host, or an admin route from a non-loopback peer).",
                  "Use the loopback address on the machine that runs the server, or add the name to the configuration on purpose."),
    "not_found": ("No such job or upload, it belongs to another token, or its result expired (results are kept `result_ttl_s`).",
                  "Check the id and the token; submit the job again if the result expired."),
    "job_not_ready": ("A result was asked for a job that has not succeeded (still running, or failed, or cancelled).",
                      "Wait for status `succeeded` (wait_job); a failed or cancelled job never has a result."),
    "model_not_installed": ("The model is not installed on this server, or its worker environment is missing.",
                            "On the server: `aha-open models install <model> --accept-license <id>`; then `aha-open models list`."),
    "unsupported_hardware": ("This machine cannot run the model (not enough VRAM, no GPU, or too little RAM).",
                             "Use a smaller model or size, or a machine that meets the requirement in docs/05_models.md."),
    "busy": ("The client's or the server's queue is full.",
             "Wait `retry_after_s` and submit again; do not loop faster than that."),
    "model_loading": ("The model is being loaded onto the GPU (tens of seconds to minutes, measured ~90 s cold for Qwen INT8).",
                      "Wait `retry_after_s`, then submit; GET /v1/manifest shows state `loading` until it is ready."),
    "gpu_out_of_memory": ("The GPU ran out of VRAM while generating.",
                          "Use a smaller size (the `1k` class), close other GPU programs, then submit again. It is not retried automatically."),
    "user_cancelled": ("The job was cancelled by a client. A cancelled job never has a result.",
                       "Nothing to fix; submit again if it was not intended."),
    "generation_failed": ("The pipeline failed for a reason that is not memory. The message says which kind (see below).",
                          "Read the message and the worker log; the cause patterns below say what to do."),
}

# `generation_failed` is one code for many causes; the gateway puts the specifics in the message. Matching is by substring
# of text the gateway itself writes (aha_open/scheduler.py, aha_open/worker_proc.py), and each pattern has a test that provokes it.
GENERATION_FAILED_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("ran out of memory", "The worker process ran out of system memory (RAM or the Windows commit limit), not VRAM.",
     "Free memory (close other programs) and try again; the Qwen INT8 worker needs about 17.6 GiB of RAM when loading."),
    ("crashed in native code", "The worker process died in native code (an access violation or a segmentation fault). The request is not the cause.",
     "Read the last stage line and the Python stack in the worker log below; try again, and if it repeats report it with that log."),
    ("could not start", "The worker could not even start (a missing DLL or a broken environment).",
     "Reinstall the model's worker environment: `aha-open models install <model> ...`."),
    ("output is ", "The worker returned an image of another size than the one accepted; the gateway refuses to resize silently.",
     "This is a worker defect, not a client mistake: report the model, the size and the worker log."),
    ("is not a PNG", "The worker wrote a file that is not a PNG.", "Report it with the worker log: the worker is broken."),
    ("timed out", "Loading the model took longer than `worker_load_timeout_s`.",
     "Look at the worker log's last stage line to see where it stalled; raise the limit only if the stage is still progressing."),
    ("exceeded job_timeout_s", "The job ran longer than `job_timeout_s`.", "Use a smaller size or fewer steps, or raise the limit on purpose."),
    ("worker is not reachable", "The gateway could not talk to the worker (it exited between jobs).", "Submit again: the lane starts a new worker."),
    ("while generating", "The worker failed during generation (the message after the colon is the worker's own).",
     "Read the worker log's last stage line and stack."),
    ("while loading the model", "The worker failed while loading the model.", "Read the worker log's last stage line and stack."),
)


def codes_in_gateway_source() -> set[str]:
    """Every error code the gateway can emit, read from its source: `ApiError(status, "code", ...)`, `LaneError("code", ...)`, and
    the worker codes the scheduler accepts."""
    found: set[str] = set(WORKER_ERROR_CODES)
    for path in Path(aha_open.__file__).resolve().parent.rglob("*.py"):
        if "workers" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            index = {"ApiError": 1, "LaneError": 0}.get(name)
            if index is not None and len(node.args) > index and isinstance(node.args[index], ast.Constant) and isinstance(node.args[index].value, str):
                found.add(node.args[index].value)
    return found


def codes_in_specification() -> set[str]:
    text = (REPO / "docs" / "03_api_v1.md").read_text(encoding="utf-8")
    section = text.split("## 8. Error codes", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `([a-z_]+)` \|", section, re.M))


def explain(code: str | None, message: str | None) -> tuple[str, str]:
    """(cause, action) for a failed job's error."""
    if not code:
        return ("No error code on this job.", "If the job failed, read the logs below.")
    cause, action = ERROR_CODE_GUIDE.get(
        code, (f"Unknown error code {code!r}: the guide does not know it.", "Read the message and the logs; this code needs an entry in guide.py.")
    )
    if code == "generation_failed" and message:
        lowered = message.lower()
        for needle, why, todo in GENERATION_FAILED_PATTERNS:
            if needle in lowered:
                return why, todo
    return cause, action
