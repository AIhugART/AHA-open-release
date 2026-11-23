"""LoRA registry: what the operator may put on the server and what is refused (plan R23). No GPU, no torch."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from aha_open import loras
from aha_open.catalog import CATALOG, validate_job
from aha_open.cli import main
from aha_open.config import Config
from aha_open.errors import ApiError

QWEN = CATALOG["qwen-image-2.1"]


def make_safetensors(path: Path, names=("transformer.blocks.0.attn.to_q.lora_A.weight", "transformer.blocks.0.attn.to_q.lora_B.weight")) -> Path:
    header = {n: {"dtype": "F32", "shape": [2, 2], "data_offsets": [i * 16, (i + 1) * 16]} for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 16 * len(names))
    return path


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(data_dir=tmp_path / "data")


def put(cfg: Config, name: str, **meta) -> Path:
    folder = loras.lora_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    path = make_safetensors(folder / f"{name}.safetensors")
    if meta:
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_scan_offers_valid_loras_and_never_exposes_paths(cfg):
    put(cfg, "ink-style", display_name="Ink style", default_scale=0.8, trigger="inkpaint", license="CC-BY-4.0")
    found, problems = loras.scan(cfg)
    assert problems == [] and set(found) == {"ink-style"}
    view = found["ink-style"].public()
    assert view["display_name"] == "Ink style" and view["default_scale"] == 0.8 and view["trigger"] == "inkpaint"
    assert "path" not in json.dumps(view) and str(cfg.data_dir) not in json.dumps(view)


def test_files_that_are_not_safe_loras_are_skipped_with_a_reason(cfg):
    folder = loras.lora_dir(cfg)
    folder.mkdir(parents=True)
    (folder / "pickled.ckpt").write_bytes(b"pickle")
    (folder / "weights.pt").write_bytes(b"pickle")
    (folder / "not-safetensors.safetensors").write_bytes(b"hello world, definitely not a header")
    make_safetensors(folder / "no-lora-tensors.safetensors", names=("model.embed.weight",))
    make_safetensors(folder / "Bad Name.safetensors")
    put(cfg, "bad-scale", default_scale=9)
    put(cfg, "good")
    found, problems = loras.scan(cfg)
    assert set(found) == {"good"}
    text = "\n".join(problems)
    assert text.count("only .safetensors files are accepted") == 2  # the extension rule itself, not a later header check
    for needle in ("pickled.ckpt", "weights.pt", "not a safetensors file", "no LoRA tensors", "Bad Name", "default_scale"):
        assert needle in text, needle


def test_a_truncated_header_cannot_make_scan_read_gigabytes(cfg):
    folder = loras.lora_dir(cfg)
    folder.mkdir(parents=True)
    (folder / "evil.safetensors").write_bytes(struct.pack("<Q", 2**40) + b"{}")  # claims a 1 TiB header
    found, problems = loras.scan(cfg)
    assert not found and "bad header" in problems[0]


def test_add_validates_copies_atomically_and_refuses_duplicates(cfg, tmp_path):
    src = make_safetensors(tmp_path / "My Style.safetensors")
    lora = loras.add(cfg, src, display_name="Mine", default_scale=0.7)
    assert lora.id == "my-style" and lora.default_scale == 0.7 and lora.path.exists()
    assert not list(loras.lora_dir(cfg).glob("*.tmp"))  # no half-copied file is left to be offered
    with pytest.raises(loras.LoraError, match="already exists"):
        loras.add(cfg, src)
    (tmp_path / "x.ckpt").write_bytes(b"pickle")
    with pytest.raises(loras.LoraError, match="only .safetensors"):
        loras.add(cfg, tmp_path / "x.ckpt")
    with pytest.raises(loras.LoraError, match="not allowed"):
        loras.add(cfg, src, "../escape")
    assert loras.remove(cfg, "my-style") and not loras.remove(cfg, "my-style") and not loras.remove(cfg, "../x")


def test_a_hash_is_cached_and_changes_when_the_file_changes(cfg):
    path = put(cfg, "a")
    first = loras.scan(cfg)[0]["a"].sha256()
    assert loras.scan(cfg)[0]["a"].sha256() == first
    make_safetensors(path, names=("x.lora_A.weight", "x.lora_B.weight", "y.lora_A.weight"))
    assert loras.scan(cfg)[0]["a"].sha256() != first


# ---------------------------------------------------------------------------------------------- request validation


def check(params, available=("ink",), variant="int8"):
    return validate_job(QWEN, "image.generate", "a cat", {"aspect_ratio": "1:1", **params}, [], frozenset(available), variant)


def test_loras_param_is_normalised_and_checked_before_queueing():
    assert check({"loras": [{"id": "ink"}]})["loras"] == [{"id": "ink", "scale": 1.0}]
    assert check({"loras": [{"id": "ink", "scale": 0.5}]})["loras"] == [{"id": "ink", "scale": 0.5}]
    for bad in (
        [{"id": "missing"}], [{"id": "ink"}, {"id": "ink"}], [{"id": "ink", "scale": 0}], [{"id": "ink", "scale": 2.5}],
        [{"id": "ink", "scale": "high"}], [{"id": "ink", "scale": True}], [{"id": "ink", "path": "/etc/passwd"}],
        [{"id": 7}], "ink", [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
    ):
        with pytest.raises(ApiError) as err:
            check({"loras": bad}, available=("ink", "a", "b", "c", "d"))
        assert err.value.status == 400, bad


def test_a_model_without_lora_support_refuses_them():
    no_loras = type(QWEN)(**{**QWEN.__dict__, "capabilities": {k: v for k, v in QWEN.capabilities.items() if k != "loras"}})
    with pytest.raises(ApiError, match="does not accept LoRAs"):
        validate_job(no_loras, "image.generate", "x", {"loras": [{"id": "ink"}], "aspect_ratio": "1:1"}, [], frozenset({"ink"}))
    with pytest.raises(ApiError, match="unknown params"):  # a video model does not even have the parameter
        validate_job(CATALOG["example-video"], "video.generate", "x", {"loras": [{"id": "ink"}]}, [], frozenset({"ink"}))


def test_a_variant_known_not_to_support_loras_refuses_them(monkeypatch):
    patched = {**QWEN.variants, "int8": {**QWEN.variants["int8"], "loras": False}}
    spec = type(QWEN)(**{**QWEN.__dict__, "variants": patched})
    with pytest.raises(ApiError, match="not supported with the int8 variant"):
        validate_job(spec, "image.generate", "x", {"loras": [{"id": "ink"}], "aspect_ratio": "1:1"}, [], frozenset({"ink"}), "int8")


def test_the_worker_never_lets_diffusers_fall_back_to_pickle():
    """diffusers' loader falls back to pickle when given a file path. The worker hands it a state dict read by
    `safetensors.torch.load_file` and never calls `torch.load` or `pickle.load`."""
    import ast

    tree = ast.parse((Path(__file__).resolve().parents[1] / "workers" / "qwen_image" / "worker.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    loads = [c for c in calls if getattr(c.func, "attr", "") == "load_lora_weights"]
    assert loads, "the worker no longer loads LoRAs through load_lora_weights"
    for call in loads:  # a dict variable, never a path: a path lets diffusers choose a pickle loader
        assert isinstance(call.args[0], ast.Name) and call.args[0].id == "state"
    assert any(isinstance(c.func, ast.Name) and c.func.id == "load_file" for c in calls), "LoRA files must be read with safetensors"
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "load" and isinstance(call.func.value, ast.Name):
            assert call.func.value.id not in ("torch", "pickle"), "the worker must not unpickle anything"


# ---------------------------------------------------------------------------------------------- CLI


def test_cli_add_list_remove(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "init"]) == 0
    src = make_safetensors(tmp_path / "paper.safetensors")
    assert main(["--data-dir", str(tmp_path), "loras", "add", str(src), "--name", "Paper", "--scale", "0.6", "--trigger", "papercut"]) == 0
    capsys.readouterr()
    assert main(["--data-dir", str(tmp_path), "loras", "list"]) == 0
    out = capsys.readouterr().out
    assert "paper" in out and "0.6" in out and "Paper" in out
    assert main(["--data-dir", str(tmp_path), "loras", "add", str(tmp_path / "missing.safetensors")]) == 2
    assert main(["--data-dir", str(tmp_path), "loras", "remove", "paper"]) == 0
    capsys.readouterr()
    assert main(["--data-dir", str(tmp_path), "loras", "list"]) == 0
    assert "none yet" in capsys.readouterr().out


def test_default_id_is_made_from_a_real_world_file_name(cfg, tmp_path):
    """Trained LoRAs are named like `qwen-image-2.1-restore.safetensors`; the dots must not make `add` need --id."""
    src = make_safetensors(tmp_path / "Qwen-Image-2.1_Restore V2.safetensors")
    assert loras.add(cfg, src).id == "qwen-image-2-1_restore-v2"
