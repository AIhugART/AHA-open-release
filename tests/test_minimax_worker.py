"""The pure parts of the MiniMax H3 worker's loader, checked in the worker's own environment (they need torch and PyAV).

Skipped when workers/minimax_h3/.venv is not installed, like the Qwen worker's tests. No GPU and no model files are needed: every
file here is made on the fly, with the shape of the defect each test guards (docs/product/05_models.md section 2.0, plan section 3.2).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "workers" / "minimax_h3"
PYTHON = WORKER_DIR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

pytestmark = pytest.mark.skipif(not PYTHON.exists(), reason="the minimax_h3 worker environment is not installed")


def run_probe(source: str, *args: str) -> dict:
    """Run ``source`` in the worker's environment with the worker folder on sys.path; it prints one JSON object."""
    code = "import json, sys\nsys.path.insert(0, sys.argv[1])\n" + source
    done = subprocess.run([str(PYTHON), "-c", code, str(WORKER_DIR), *args], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------------------------- the trailer


TRAILER = r"""
import struct, tempfile, os, torch
from lenient_safetensors import SafetensorsFile

def make(path, trailer):
    a = torch.arange(6, dtype=torch.float32).numpy().tobytes()
    b = torch.arange(4, dtype=torch.int8).numpy().tobytes()
    header = {"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, len(a)]},
              "b": {"dtype": "I8", "shape": [4], "data_offsets": [len(a), len(a) + len(b)]},
              "__metadata__": {"k": "v"}}
    h = json.dumps(header).encode()
    open(path, "wb").write(struct.pack("<Q", len(h)) + h + a + b + trailer)

d = tempfile.mkdtemp()
out = {}
make(d + "/plain.safetensors", b"")
make(d + "/trailer.safetensors", b"\nL2P_bypass_x.safetensors_1785751234\n")
for name in ("plain", "trailer"):
    with SafetensorsFile(d + f"/{name}.safetensors") as f:
        out[name] = {"keys": sorted(f.keys()), "a": f.get("a").tolist(), "b": f.get("b").tolist(), "meta": f.metadata}
open(d + "/cut.safetensors", "wb").write(open(d + "/plain.safetensors", "rb").read()[:-3])
try:
    SafetensorsFile(d + "/cut.safetensors"); out["cut"] = "accepted"
except ValueError as e:
    out["cut"] = str(e)
try:
    from safetensors import safe_open
    safe_open(d + "/trailer.safetensors", "pt"); out["std_reader_accepts_trailer"] = True
except Exception:
    out["std_reader_accepts_trailer"] = False
print(json.dumps(out))
"""


def test_a_trailer_after_the_last_tensor_is_tolerated_and_the_values_are_exact():
    out = run_probe(TRAILER)
    for name in ("plain", "trailer"):
        assert out[name]["keys"] == ["a", "b"] and out[name]["meta"] == {"k": "v"}
        assert out[name]["a"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]] and out[name]["b"] == [0, 1, 2, 3]
    assert out["std_reader_accepts_trailer"] is False  # the reason this reader exists: safetensors 0.8 refuses these files


def test_a_file_cut_short_is_refused_not_read_as_garbage():
    out = run_probe(TRAILER)
    assert "truncated" in out["cut"]


# ---------------------------------------------------------------------------------------------- INT4 and the rotation


NUMERICS = r"""
import torch, convrot

out = {}
# every pair of signed nibbles, low nibble first
values = torch.tensor([[v0, v1] for v0 in range(-8, 8) for v1 in range(-8, 8)], dtype=torch.int8)
packed = ((values[:, 0] & 0xF) | ((values[:, 1] & 0xF) << 4)).to(torch.uint8).view(-1, 1)
out["unpack_exact"] = bool(torch.equal(convrot.unpack_int4(packed).reshape(-1, 2), values))

h = convrot.hadamard(256, torch.device("cpu"))
out["hadamard_orthogonal"] = float((h @ h.T - torch.eye(256)).abs().max())
out["hadamard_symmetric"] = bool(torch.allclose(h, h.T))
try:
    convrot.hadamard(200, torch.device("cpu")); out["bad_size"] = "accepted"
except ValueError:
    out["bad_size"] = "refused"

# dequantize: stored = round((W @ H) / scale) per 256-group; decoding must give W back up to the rounding of the INT8 grid
torch.manual_seed(0)
W = torch.randn(32, 512)
rot = (W.view(32, 2, 256) @ h).view(32, 512)
scale = rot.abs().amax(dim=1) / 127
q = torch.round(rot / scale[:, None]).to(torch.int8)
back = convrot.dequantize(q, scale, torch.float32)
out["roundtrip_rel_err"] = float((back - W).norm() / W.norm())
out["wrong_scale_rel_err"] = float((convrot.dequantize(q, scale * 2, torch.float32) - W).norm() / W.norm())
print(json.dumps(out))
"""


def test_int4_unpack_is_exact_for_every_nibble_pair():
    assert run_probe(NUMERICS)["unpack_exact"] is True


def test_the_hadamard_matrix_is_its_own_inverse_and_only_powers_of_four_are_accepted():
    out = run_probe(NUMERICS)
    assert out["hadamard_orthogonal"] < 1e-5 and out["hadamard_symmetric"] is True and out["bad_size"] == "refused"


def test_decoding_a_rotated_int8_weight_returns_the_original_weight():
    out = run_probe(NUMERICS)
    assert out["roundtrip_rel_err"] < 0.01  # only the 8-bit grid is lost
    assert out["wrong_scale_rel_err"] > 0.5  # and the check is not blind: a wrong scale is far off


# ---------------------------------------------------------------------------------------------- the time table


TIME = r"""
import torch
from h3_models import BakedTimeEmbedder

table = torch.arange(1025, dtype=torch.float32).unsqueeze(1).repeat(1, 8) * torch.arange(1, 9, dtype=torch.float32)  # row i = i * (1..8)
e = BakedTimeEmbedder(table)
t = torch.tensor([0.0, 1.0, 0.5, 1 / 1024, 0.25 + 0.5 / 1024, -0.3, 1.7])
out = {"first_col": e(t)[:, 0].tolist(), "shape": list(e(t).shape), "last_col": e(t)[:, 7].tolist()}
print(json.dumps(out))
"""


def test_the_time_table_is_read_at_t_times_1024_with_linear_interpolation_and_clamped_ends():
    out = run_probe(TIME)
    first = out["first_col"]  # row i holds the value i, so the lookup returns i = t * 1024
    assert out["shape"] == [7, 8]
    assert first[0] == 0.0 and first[1] == 1024.0 and first[2] == 512.0  # ends and the middle
    assert first[3] == pytest.approx(1.0)  # one grid step
    assert first[4] == pytest.approx(256.5, abs=1e-3)  # between two rows: halfway
    assert first[5] == 0.0 and first[6] == 1024.0  # outside [0, 1] stays on the table
    assert out["last_col"][2] == pytest.approx(512.0 * 8)  # every column follows its own row


# ---------------------------------------------------------------------------------------------- key mapping


KEYS = r"""
from h3_models import _rename
cases = ["blocks.3.attn.q_norm.weight", "blocks.3.attn.out_proj.weight", "blocks.3.mlp.fc2.weight", "blocks.3.adaln_proj.linear.bias",
         "token_refiner.blocks.1.attn.k_norm.weight", "token_refiner.final_norm.weight", "video_patch_proj.weight", "audio_patch_proj.bias",
         "condition_proj.weight", "final_layer.norm.weight", "final_layer.adaln_proj.linear.weight", "final_layer.video_out.bias",
         "final_layer.audio_out.weight", "blocks.3.norm1.weight"]
print(json.dumps({c: _rename(c) for c in cases}))
"""


def test_names_in_the_files_map_onto_the_diffusers_modules():
    out = run_probe(KEYS)
    assert out["blocks.3.attn.q_norm.weight"] == "transformer_blocks.3.attn.norm_q.weight"
    assert out["blocks.3.attn.out_proj.weight"] == "transformer_blocks.3.attn.to_out.0.weight"
    assert out["blocks.3.mlp.fc2.weight"] == "transformer_blocks.3.ff.net.2.weight"
    assert out["blocks.3.adaln_proj.linear.bias"] == "transformer_blocks.3.adaln_proj.linear.bias"
    assert out["token_refiner.blocks.1.attn.k_norm.weight"] == "token_refiner.refiner_blocks.1.attn.norm_k.weight"
    assert out["token_refiner.final_norm.weight"] == "token_refiner.final_norm.weight"
    assert out["video_patch_proj.weight"] == "proj_in.weight" and out["audio_patch_proj.bias"] == "audio_proj_in.bias"
    assert out["condition_proj.weight"] == "context_embedder.weight"
    assert out["final_layer.norm.weight"] == "norm_out.norm.weight"
    assert out["final_layer.adaln_proj.linear.weight"] == "norm_out.linear.weight"
    assert out["final_layer.video_out.bias"] == "proj_out.bias" and out["final_layer.audio_out.weight"] == "audio_proj_out.weight"
    assert out["blocks.3.norm1.weight"] == "transformer_blocks.3.norm1.weight"


MODEL_COVERAGE = r"""
import torch
from diffusers import MiniMaxH3Transformer3DModel
from h3_models import _rename

cfg = json.loads(open(sys.argv[2]).read())
cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
cfg["time_embed_dim"] = 8
with torch.device("meta"):
    model = MiniMaxH3Transformer3DModel(**cfg)
have = set(model.state_dict())
# every name the community files use for a block-0 / refiner-0 / top-level tensor, from the real header (see docs/product/05_models.md section 2.0)
src = ["blocks.0.adaln_proj.linear.bias", "blocks.0.adaln_proj.linear.weight", "blocks.0.attn.k_norm.weight", "blocks.0.attn.q_norm.weight",
       "blocks.0.attn.out_proj.weight", "blocks.0.mlp.fc2.weight", "blocks.0.norm1.weight", "blocks.0.norm2.weight",
       "token_refiner.blocks.0.attn.k_norm.weight", "token_refiner.blocks.0.attn.q_norm.weight", "token_refiner.blocks.0.attn.out_proj.weight",
       "token_refiner.blocks.0.mlp.fc2.weight", "token_refiner.blocks.0.norm1.weight", "token_refiner.blocks.0.norm2.weight",
       "token_refiner.final_norm.weight", "audio_patch_proj.bias", "audio_patch_proj.weight", "video_patch_proj.bias", "video_patch_proj.weight",
       "condition_proj.bias", "condition_proj.weight", "final_layer.adaln_proj.linear.bias", "final_layer.adaln_proj.linear.weight",
       "final_layer.audio_out.bias", "final_layer.audio_out.weight", "final_layer.norm.weight", "final_layer.video_out.bias",
       "final_layer.video_out.weight"]
print(json.dumps({"missing": [k for k in src if _rename(k) not in have]}))
"""

OFFICIAL_CONFIG = ROOT / "reference" / "h3_weights" / "official" / "transformer" / "config.json"


@pytest.mark.skipif(not OFFICIAL_CONFIG.exists(), reason="the official configuration files are not downloaded")
def test_every_renamed_name_exists_in_the_diffusers_model():
    out = run_probe(MODEL_COVERAGE, str(OFFICIAL_CONFIG))
    assert out["missing"] == []  # a rename that points nowhere would leave a tensor with no place (the loader refuses that too)


# ---------------------------------------------------------------------------------------------- the mp4


VIDEO = r"""
import numpy as np, os, tempfile
import h3_video

rng = np.random.default_rng(1)
frames = rng.integers(0, 255, (10, 64, 96, 3), dtype=np.uint8)
audio = (rng.standard_normal((2, 32000)) * 0.05).astype(np.float32)
folder = tempfile.mkdtemp()
path = folder + "/x.mp4"
h3_video.write_mp4(path, frames, audio, 32000)
info = h3_video.probe(path)
bad = {}
for n, (label, f, a) in enumerate((("float frames", frames.astype(np.float32), audio), ("mono", frames, audio[:1]), ("wrong rank", frames[0], audio))):
    target = folder + f"/bad{n}.mp4"
    try:
        h3_video.write_mp4(target, f, a, 32000); bad[label] = "accepted"
    except ValueError:
        bad[label] = "refused"
    except Exception as e:  # any other error type means the input got past the checks
        bad[label] = "failed late: " + type(e).__name__
    bad[label] += " / file left behind" if os.path.exists(target) else " / no file"
print(json.dumps({"info": info, "bad": bad, "label": h3_video.AI_LABEL}))
"""


def test_the_mp4_holds_what_was_written_with_the_disclosure_label_and_no_prompt():
    out = run_probe(VIDEO)
    info = out["info"]
    assert (info["width"], info["height"], info["frames"], info["fps"]) == (96, 64, 10, 24.0)
    assert info["audio_rate"] == 32000 and info["audio_channels"] == 2 and info["codec"] == "h264"
    assert info["comment"] == out["label"] == "AI-generated (MiniMax H3)"


def test_the_writer_refuses_input_of_the_wrong_shape_before_it_creates_any_file():
    # Mono audio is also refused by PyAV, but only while encoding, after the file was opened and half written: the point of the
    # writer's own check is that nothing is left on disk, so that is what is asserted (a check that only looked at the exception would
    # pass with the writer's check removed).
    assert run_probe(VIDEO)["bad"] == {
        "float frames": "refused / no file", "mono": "refused / no file", "wrong rank": "refused / no file",
    }


# ---------------------------------------------------------------------------------------------- Ref2VA references (plan M7)


REFERENCES = r"""
import tempfile, os
from PIL import Image
import h3_references

d = tempfile.mkdtemp()
red, blue = os.path.join(d, "red.png"), os.path.join(d, "blue.png")
Image.new("RGB", (64, 48), (255, 0, 0)).save(red)
Image.new("RGB", (32, 32), (0, 0, 255)).save(blue)
out = {}

refs = h3_references.from_inputs([{"role": "reference", "path": blue}, {"role": "reference", "path": red}])
out["kinds"] = [r.kind for r in refs]
out["sizes"] = [list(r.image.size) for r in refs]  # the order the client sent is the order the model reads

def refused(inputs):
    try:
        h3_references.from_inputs(inputs)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return "ACCEPTED"

out["url"] = refused([{"role": "reference", "path": "https://example.invalid/a.png"}])
out["role"] = refused([{"role": "mask", "path": red}])
out["missing"] = refused([{"role": "reference", "path": os.path.join(d, "none.png")}])
print(json.dumps(out))
"""


def test_references_keep_the_order_the_client_sent_and_refuse_what_the_worker_must_not_do():
    out = run_probe(REFERENCES)
    assert out["kinds"] == ["image", "image"] and out["sizes"] == [[32, 32], [64, 48]]
    assert "not a URL" in out["url"] and out["url"].startswith("ValueError")  # a worker never fetches from the network
    assert "not a reference role" in out["role"]
    assert out["missing"] != "ACCEPTED"  # a missing file is an error, not an empty reference


def test_an_audio_reference_alone_is_refused_by_the_worker_too():
    out = run_probe(
        "import h3_references\n"
        "class A:\n    kind = 'audio'\n"
        "orig = h3_references.__dict__['ROLE_KIND']\n"
        "import types, sys\n"
        "from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference as M\n"
        "M.from_file = classmethod(lambda cls, p: A())\n"
        "try:\n    h3_references.from_inputs([{'role': 'reference_audio', 'path': 'x'}])\n    r = 'ACCEPTED'\n"
        "except ValueError as e:\n    r = str(e)\n"
        "print(json.dumps({'r': r}))\n"
    )
    assert "needs at least one image or video reference" in out["r"]


# ---------------------------------------------------------------------------------------------- the encoder's attention (plan M7)


ATTENTION = r"""
import torch, torch.nn.functional as F
import h3_models

class M:  # what transformers hands the attention function: the module carries the GQA group count
    num_key_value_groups = 8
    is_causal = True

torch.manual_seed(0)
q = torch.randn(1, 16, 40, 8)
k = torch.randn(1, 2, 40, 8)
v = torch.randn(1, 2, 40, 8)

def reference(mask=None, causal=False):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=True, scale=0.35)

out = {}
got, weights = h3_models._sdpa_with_repeated_kv(M(), q, k, v, None, scaling=0.35)  # no mask + causal module = causal attention
out["causal_diff"] = (got.transpose(1, 2) - reference(causal=True)).abs().max().item()
out["shape"] = list(got.shape)  # [batch, tokens, heads, head_dim], what the decoder layer expects
out["weights"] = weights
mask = torch.rand(1, 1, 40, 40) > 0.3
mask |= torch.eye(40, dtype=torch.bool)[None, None]  # every row keeps at least itself
got, _ = h3_models._sdpa_with_repeated_kv(M(), q, k, v, mask, scaling=0.35)
out["mask_diff"] = (got.transpose(1, 2) - reference(mask=mask)).abs().max().item()

class Vision:  # the vision tower has no GQA groups and says it is not causal
    is_causal = False
qv = torch.randn(1, 4, 12, 8)
got, _ = h3_models._sdpa_with_repeated_kv(Vision(), qv, qv, qv, None, scaling=0.35, is_causal=False)
out["vision_diff"] = (got.transpose(1, 2) - F.scaled_dot_product_attention(qv, qv, qv, scale=0.35)).abs().max().item()
print(json.dumps(out))
"""


def test_the_attention_without_gqa_flag_gives_the_same_numbers_as_sdpa_with_it():
    out = run_probe(ATTENTION)
    assert out["shape"] == [1, 40, 16, 8] and out["weights"] is None
    assert out["causal_diff"] < 1e-5 and out["mask_diff"] < 1e-5 and out["vision_diff"] < 1e-5


def test_building_the_encoder_installs_that_attention_and_rebuilds_the_vision_rotary_table():
    """Source gate (AST, not text): without the first, a reference image needs a 10 GiB score matrix and the job dies out of memory;
    without the second, the vision tower stays on the meta device and the job fails before it starts (both seen on 2026-11-25)."""
    import ast

    tree = ast.parse((WORKER_DIR / "h3_models.py").read_text(encoding="utf-8"))
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_text_encoder")
    registers = [
        n for n in ast.walk(build) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "register"
        and [getattr(a, "value", getattr(a, "id", None)) for a in n.args] == ["sdpa", "_sdpa_with_repeated_kv"]
    ]
    assert registers, "build_text_encoder must register _sdpa_with_repeated_kv as the 'sdpa' attention"
    rotary = [
        n for n in ast.walk(build) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "rotary_pos_emb" for t in n.targets)
    ]
    assert rotary, "build_text_encoder must rebuild visual.rotary_pos_emb when the vision tower is loaded"


NVFP4 = r"""
import torch, comfy_kitchen as ck
from convrot import Nvfp4Linear, QUANTIZED_LINEARS

torch.manual_seed(0)
rows, cols = 256, 64
x = torch.randn(rows, cols)
tensor_scale = torch.tensor(x.abs().max().item() / (448 * 6))
packed, tiled = ck.quantize_nvfp4(x, tensor_scale)            # comfy-kitchen packs and tiles; it is the reference
reference = ck.dequantize_nvfp4(packed, tensor_scale, tiled, torch.float32)
layer = Nvfp4Linear.from_file(packed, tiled, tensor_scale)
mine = layer.dense(torch.float32)
y = torch.randn(3, cols)
print(json.dumps({
    "vs_reference": (mine - reference).abs().max().item(),
    "noise": ((mine - x).norm() / x.norm()).item(),
    "rows": torch.equal(layer.rows(64, 128).dense(torch.float32), mine[64:128]),
    "swapped": torch.equal(layer.swapped(128).dense(torch.float32), torch.cat([mine[128:], mine[:128]])),
    "forward": (layer(y) - y @ mine.T).abs().max().item(),
    "listed": isinstance(layer, QUANTIZED_LINEARS),
    "scale_bytes": layer.block_scale.element_size(),
}))
"""


def test_nvfp4_decodes_like_comfy_kitchen_and_its_rows_can_be_sliced():
    out = run_probe(NVFP4)
    assert out["vs_reference"] == 0.0 and out["noise"] < 0.2
    assert out["rows"] and out["swapped"] and out["listed"] and out["forward"] < 1e-5
    assert out["scale_bytes"] == 1  # float32 scales cost 4 GiB of GPU memory on the real model
