"""Checks that need the real files (plan D5): they skip when the files are not on this machine.

D5: the community ConvRot file stores the feed-forward's two input layers fused as ``img_mlp.gate_up`` while diffusers has
``gate_layer`` and ``proj``. Which half is which was proved by comparing the two halves, decoded, with the bitsandbytes build of
the same BF16 model - not assumed from a third party's statement. The same fact is what ``lora_state_for`` relies on for ai-toolkit
LoRAs that fuse the two layers too.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "workers" / "qwen_image"
PYTHON = WORKER_DIR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
DATA = Path(os.environ.get("AHA_OPEN_DATA_DIR") or (Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AHA-open"))
BNB = DATA / "models" / "qwen-image-2.1" / "int8" / "transformer"
CONVROT = DATA / "models" / "qwen-image-2.1" / "convrot" / "transformers" / "qwen_image_2.1_bf16_int8_convrot.safetensors"

PROBE = r"""
import json, sys, glob
sys.path.insert(0, sys.argv[1])
import torch
from safetensors import safe_open
from lenient_safetensors import SafetensorsFile
from convrot import dequantize

bnb_dir, convrot_file = sys.argv[2], sys.argv[3]
prefix = "transformer_blocks.0.img_mlp."
cr = SafetensorsFile(convrot_file)
w = dequantize(cr.get(prefix + "gate_up.weight"), cr.get(prefix + "gate_up.weight_scale"), torch.float32).double()
half = w.shape[0] // 2
tensors = {}
for shard in sorted(glob.glob(bnb_dir + "/*.safetensors")):
    with safe_open(shard, "pt") as f:
        for key in f.keys():
            if key.startswith(prefix) and key.endswith((".weight", ".SCB")):
                tensors[key] = f.get_tensor(key)
def bnb(name):
    return tensors[prefix + name + ".weight"].double() * tensors[prefix + name + ".SCB"].double().reshape(-1, 1) / 127.0
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
gate, proj = bnb("gate_layer"), bnb("proj")
print(json.dumps({
    "first_gate": cos(w[:half], gate), "first_proj": cos(w[:half], proj),
    "second_gate": cos(w[half:], gate), "second_proj": cos(w[half:], proj),
}))
"""


@pytest.mark.skipif(not (PYTHON.exists() and BNB.is_dir() and CONVROT.is_file()), reason="needs the worker environment and both real builds")
def test_gate_up_is_gate_layer_then_proj():
    done = subprocess.run([str(PYTHON), "-I", "-c", PROBE, str(WORKER_DIR), str(BNB), str(CONVROT)], capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr[-1500:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    # the same weights quantized two ways: the matching half is almost the same vector, the other one is unrelated
    assert out["first_gate"] > 0.99 and out["second_proj"] > 0.99, out
    assert abs(out["first_proj"]) < 0.1 and abs(out["second_gate"]) < 0.1, out


def test_the_loader_splits_gate_up_in_the_order_that_was_measured():
    """The default and the assignment in qwen_models.py must say gate_layer first (and LoRA's lora_state_for says the same)."""
    source = (WORKER_DIR / "qwen_models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_transformer")
    default = dict(zip([a.arg for a in func.args.args][-1:], func.args.defaults[-1:], strict=False))["gate_first"]
    assert isinstance(default, ast.Constant) and default.value is True
    assert '("gate_layer", "proj") if gate_first' in source
    worker = (WORKER_DIR / "worker.py").read_text(encoding="utf-8")
    assert 'targets = {prefix + "gate_layer": b[:half], prefix + "proj": b[half:]}' in worker
