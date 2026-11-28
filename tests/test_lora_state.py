"""The Qwen worker must apply a LoRA completely or refuse it (docs/product/05_models.md section 3.2).

Root cause this guards: a real ai-toolkit LoRA fuses two layers into `img_mlp.gate_up`; peft dropped those tensors and
only logged it, so the LoRA ran at five sixths strength with no error. `lora_state_for` runs in the worker's own
environment (it needs torch), so these tests run it there and skip when that environment is not installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "workers" / "qwen_image"
PYTHON = WORKER_DIR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

pytestmark = pytest.mark.skipif(not PYTHON.exists(), reason="the qwen worker environment is not installed")

PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch
from safetensors.torch import save_file
import worker

kind, path = sys.argv[2], sys.argv[3]
r, d, h = 4, 8, 12
pre = "diffusion_model.transformer_blocks.0."
tensors = {}
if kind in ("fused", "fused_alpha"):
    tensors[pre + "img_mlp.gate_up.lora_A.weight"] = torch.randn(r, d)
    tensors[pre + "img_mlp.gate_up.lora_B.weight"] = torch.arange(2 * h * r, dtype=torch.float32).reshape(2 * h, r)
    tensors[pre + "attn.to_q.lora_A.weight"] = torch.randn(r, d)
    tensors[pre + "attn.to_q.lora_B.weight"] = torch.randn(d, r)
    if kind == "fused_alpha":
        tensors[pre + "attn.to_q.alpha"] = torch.tensor(2.0)
elif kind == "foreign":
    tensors["transformer_blocks.0.attn.not_a_layer.lora_A.weight"] = torch.randn(r, d)
    tensors["transformer_blocks.0.attn.not_a_layer.lora_B.weight"] = torch.randn(d, r)
elif kind == "plainweight":
    tensors["transformer_blocks.0.attn.to_q.weight"] = torch.randn(d, d)
save_file(tensors, path)
names = {"transformer_blocks.0.attn.to_q", "transformer_blocks.0.img_mlp.gate_layer", "transformer_blocks.0.img_mlp.proj"}
try:
    state = worker.lora_state_for(path, names)
except ValueError as err:
    print(json.dumps({"error": str(err)}))
else:
    out = {k: list(v.shape) for k, v in state.items()}
    if kind.startswith("fused"):
        fused = tensors[pre + "img_mlp.gate_up.lora_B.weight"]
        out["gate_is_first_half"] = bool(torch.equal(state["transformer.transformer_blocks.0.img_mlp.gate_layer.lora_B.weight"], fused[:h]))
        out["proj_is_second_half"] = bool(torch.equal(state["transformer.transformer_blocks.0.img_mlp.proj.lora_B.weight"], fused[h:]))
        if kind == "fused_alpha":
            q = state["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"]
            out["q_scaled"] = bool(torch.allclose(q, tensors[pre + "attn.to_q.lora_B.weight"] * (2.0 / r)))
    print(json.dumps(out))
"""


def probe(kind: str, tmp_path: Path) -> dict:
    done = subprocess.run(
        [str(PYTHON), "-I", "-c", PROBE, str(WORKER_DIR), kind, str(tmp_path / f"{kind}.safetensors")],
        capture_output=True, text=True, check=False, timeout=180,
    )
    assert done.returncode == 0, done.stderr[-800:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_fused_gate_up_is_split_into_the_model_layers(tmp_path):
    out = probe("fused", tmp_path)
    assert out["transformer.transformer_blocks.0.img_mlp.gate_layer.lora_B.weight"] == [12, 4]
    assert out["transformer.transformer_blocks.0.img_mlp.proj.lora_B.weight"] == [12, 4]
    assert out["gate_is_first_half"] and out["proj_is_second_half"]
    assert not any("gate_up" in key for key in out)


def test_alpha_is_folded_into_the_weights(tmp_path):
    assert probe("fused_alpha", tmp_path)["q_scaled"] is True


def test_a_lora_for_another_model_is_refused_not_half_applied(tmp_path):
    assert "does not have" in probe("foreign", tmp_path)["error"]


def test_a_non_lora_tensor_is_refused(tmp_path):
    assert "not a PEFT/ai-toolkit" in probe("plainweight", tmp_path)["error"]
