"""The ConvRot half of the Qwen worker: the layer math, the LoRA path, and the places where it must agree with the gateway.

The layer and LoRA tests run in the worker's own environment (they need torch) and skip when it is not installed. They use tiny
layers made on the fly, each with the shape of the defect it guards: no GPU and no model file is needed.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aha_open.catalog import CATALOG
from aha_open.install import INSTALLABLE

ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "workers" / "qwen_image"
PYTHON = WORKER_DIR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
needs_env = pytest.mark.skipif(not PYTHON.exists(), reason="the qwen worker environment is not installed")
QWEN = CATALOG["qwen-image-2.1"]


def run_probe(source: str) -> dict:
    code = "import json, sys\nsys.path.insert(0, sys.argv[1])\n" + source
    done = subprocess.run([str(PYTHON), "-I", "-c", code, str(WORKER_DIR)], capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-1500:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _constant(path: Path, name: str):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path.name}")


# ---------------------------------------------------------------------------------------------- the worker agrees with the gateway


def test_the_worker_knows_exactly_the_convrot_variants_the_catalog_names():
    convrot = {name for name, info in QWEN.variants.items() if info["runtime"] == "convrot"}
    assert set(_constant(WORKER_DIR / "worker.py", "CONVROT_VARIANTS")) == convrot
    assert QWEN.variants["int8"]["runtime"] == "bnb"


def test_the_worker_looks_for_the_text_encoder_where_the_installer_puts_it():
    where = _constant(WORKER_DIR / "worker.py", "CONVROT_ENCODER")
    for variant in ("int8-convrot", "mixed-p010", "mixed-p015"):
        pinned = {
            f"{p['subdir']}/{name}" if p["subdir"] else name
            for p in INSTALLABLE["qwen-image-2.1"]["variants"][variant]["parts"] for name in p["files"]
        }
        assert where in pinned


def test_the_worker_does_not_import_the_gateway():
    """The worker runs in its own environment: nothing of aha_open may be importable from it."""
    for name in ("worker.py", "qwen_models.py", "convrot.py", "convrot_lora.py"):
        tree = ast.parse((WORKER_DIR / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not [m for m in modules if m.split(".")[0] == "aha_open"], f"{name} imports the gateway"


# ---------------------------------------------------------------------------------------------- the layer


LAYER = r"""
import torch
import convrot
torch.manual_seed(0)
N, K = 32, 512
w = torch.randint(-127, 128, (N, K), dtype=torch.int8)
s = torch.rand(N) * 0.01
x = torch.randn(3, 20, K, dtype=torch.bfloat16)
ref = torch.nn.functional.linear(x.double(), convrot.dequantize(w, s, torch.float32).double())
out = convrot.ConvRotLinear(w, s, False)(x)
packed = torch.randint(-128, 128, (N, K // 2), dtype=torch.int8)
ref4 = torch.nn.functional.linear(x.double(), convrot.dequantize(convrot.unpack_int4(packed), s, torch.float32).double())
out4 = convrot.ConvRotLinear(packed, s, True)(x)
every = torch.arange(-128, 128, dtype=torch.int8).reshape(1, 256)
print(json.dumps({
    "shape": list(out.shape),
    "rel": float((out.double() - ref).norm() / ref.norm()),
    "rel4": float((out4.double() - ref4).norm() / ref4.norm()),
    "unpack_same": bool(convrot.unpack_int4(every).equal(convrot.unpack_int4_fast(every))),
}))
"""


@needs_env
def test_the_rotated_activation_layer_matches_the_float64_reference():
    """x @ W.T == (x @ H) @ S.T for the stored S = W @ H; bf16-sized error for the INT8 layer and for the packed INT4 one."""
    out = run_probe(LAYER)
    assert out["shape"] == [3, 20, 32]
    assert out["rel"] < 0.02 and out["rel4"] < 0.02
    assert out["unpack_same"]  # the fast nibble unpack agrees with the plain one on every byte value


SLICES = r"""
import convrot
rows = [17, 100, 4096, 4097, 4109, 4112, 4113, 8192, 8205, 37000]
print(json.dumps({str(n): convrot.int8_slices(n) for n in rows}))
"""


@needs_env
def test_no_slice_of_the_integer_product_is_too_short_for_it():
    """`torch._int_mm` refuses 16 rows or fewer. A real job failed ("got 13") because only the TOTAL was checked: 4096 + 13 rows
    were cut into a full slice and a 13-row tail."""
    for rows, slices in run_probe(SLICES).items():
        assert [a for a, _ in slices] == [0, *[b for _, b in slices[:-1]]] and slices[-1][1] == int(rows)  # contiguous, complete
        assert all(16 < b - a <= 4096 + 16 for a, b in slices), (rows, slices)


# ---------------------------------------------------------------------------------------------- the LoRA path


LORA = r"""
import torch, torch.nn as nn
import convrot, convrot_lora
torch.manual_seed(1)
K, N, R = 256, 64, 4
w = torch.randint(-127, 128, (N, K), dtype=torch.int8)
conv = convrot.ConvRotLinear(w, torch.rand(N) * 0.01, False)
plain = nn.Linear(K, N, bias=False).to(torch.bfloat16)
model = nn.ModuleDict({"conv": conv, "plain": plain})
x = torch.randn(5, K, dtype=torch.bfloat16)
base = {k: model[k](x).clone() for k in model}
A = {k: torch.randn(R, K) for k in model}
B = {k: torch.randn(N, R) * 0.1 for k in model}
state = {}
for k in model:
    state[f"transformer.{k}.lora_A.weight"] = A[k]
    state[f"transformer.{k}.lora_B.weight"] = B[k]
same = lambda: all(torch.equal(model[k](x), base[k]) for k in model)
res = {}
res["layers"] = convrot_lora.attach(model, "mine", state)
res["idle_is_base"] = same()  # attached but not activated
convrot_lora.activate([{"id": "mine", "scale": 0.5}])
res["maxdiff"] = max(
    float((model[k](x).float() - (base[k].float() + 0.5 * (x.float() @ A[k].bfloat16().float().T @ B[k].bfloat16().float().T))).abs().max())
    for k in model)
res["changed"] = all(not torch.equal(model[k](x), base[k]) for k in model)
convrot_lora.activate([{"id": "mine", "scale": 0.0}])
res["zero_is_base"] = same()
convrot_lora.activate([{"id": "other", "scale": 1.0}])  # an adapter that was never attached adds nothing
res["unknown_is_base"] = same()
convrot_lora.activate([{"id": "mine", "scale": 1.0}])
convrot_lora.activate([])
res["off_is_base"] = same()
convrot_lora.activate([{"id": "mine", "scale": 1.0}])
convrot_lora.delete(model, "mine")
res["deleted_is_base"] = same() and not convrot_lora.ACTIVE
# a LoRA that does not fit one layer is refused and leaves NOTHING on the layers that did fit
bad = dict(state)
bad["transformer.plain.lora_A.weight"] = torch.randn(R, K + 8)
try:
    convrot_lora.attach(model, "bad", bad)
    res["bad"] = "accepted"
except ValueError:
    res["bad"] = "refused"
res["left_behind"] = any(hasattr(model[k], "aha_lora") and len(model[k].aha_lora) for k in model)
try:
    convrot_lora.attach(model, "ghost", {"transformer.nope.lora_A.weight": A["conv"], "transformer.nope.lora_B.weight": B["conv"]})
    res["ghost"] = "accepted"
except ValueError:
    res["ghost"] = "refused"
print(json.dumps(res))
"""


@needs_env
def test_a_lora_adds_scale_times_a_b_x_to_conv_and_plain_layers_and_nothing_when_off():
    res = run_probe(LORA)
    assert res["layers"] == 2
    assert res["idle_is_base"]  # attached adapters do nothing until a job names them
    assert res["maxdiff"] < 0.15 and res["changed"]  # bf16 arithmetic: close to base + scale * (x A^T) B^T, and clearly not the base
    assert res["zero_is_base"] and res["unknown_is_base"] and res["off_is_base"] and res["deleted_is_base"]  # bit-identical to no LoRA


@needs_env
def test_a_lora_that_does_not_fit_is_refused_whole():
    res = run_probe(LORA)
    assert res["bad"] == "refused" and res["ghost"] == "refused" and res["left_behind"] is False
