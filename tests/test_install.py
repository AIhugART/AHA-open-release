"""Variant choice, size tiers and `models install` (no GPU, no download: everything here is decidable offline)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from aha_open import catalog
from aha_open.catalog import (
    CATALOG,
    QWEN_SIZES,
    manifest_model,
    resolve_image_size,
    validate_job,
    weights_dir,
)
from aha_open.cli import main
from aha_open.config import Config, ModelConfig, load
from aha_open.errors import ApiError
from aha_open.hardware import INT8_MIN_VRAM_GIB, recommend_variant
from aha_open.install import INSTALLABLE

ROOT = Path(__file__).resolve().parents[1]
QWEN = CATALOG["qwen-image-2.1"]


# ---------------------------------------------------------------------------------------------- recommendation


@pytest.mark.parametrize(
    ("gpu_mib", "ram", "expected"),
    [
        (24564, 32.0, "int8"),  # the reference machine: RTX 3090 Ti
        (49140, 64.0, "int8"),  # a big card still gets int8: the full BF16 model is not offered
        (12288, 16.0, "int8"),
        (8192, 16.0, None),  # below what int8 was measured to need
        (None, 32.0, None),  # no GPU
        (0, 32.0, None),
    ],
)
def test_recommend_variant(gpu_mib, ram, expected):
    variant, reason = recommend_variant(gpu_mib, ram)
    assert variant == expected
    assert reason and reason[0].islower()  # one sentence the CLI can put after a colon


FOUR = ["int8", "int8-convrot", "mixed-p010", "mixed-p015"]


def test_qwen_offers_four_variants_and_never_bf16():
    """The full BF16 model was dropped: no catalog entry, install plan or worker path may bring it back."""
    assert list(QWEN.variants) == FOUR and set(QWEN.min_vram_mib) <= set(FOUR)
    assert list(INSTALLABLE["qwen-image-2.1"]["variants"]) == FOUR
    assert "bf16" not in QWEN.variants
    source = (ROOT / "workers" / "qwen_image" / "worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "BF16_RESIDENT_GIB"]
    assert "unsupported variant" in source


def test_min_vram_follows_the_measured_peak():
    assert QWEN.min_vram_mib["int8"] >= INT8_MIN_VRAM_GIB * 1024 > 9.47 * 1024  # measured int8 peak at 1024x1024


def test_a_configuration_left_over_from_the_bf16_days_says_what_to_do(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "detect_gpus", lambda ttl_s=30.0: [{"index": 0, "name": "RTX", "memory_total_mib": 49140}])
    mc = ModelConfig("qwen-image-2.1", "qwen_image", variant="bf16")
    state, detail = catalog.model_state(Config(data_dir=tmp_path, models=[mc]), mc, QWEN)
    assert state == "error" and "not offered" in detail and "models install qwen-image-2.1" in detail


# ---------------------------------------------------------------------------------------------- sizes


def test_qwen_sizes_are_valid_for_the_pipeline_and_come_in_two_tiers():
    native = [s for s in QWEN_SIZES if s["tier"] == "native"]
    small = {s["aspect_ratio"]: s for s in QWEN_SIZES if s["tier"] == "1k"}
    assert len(native) == len(small) == 7
    assert len({(s["width"], s["height"]) for s in QWEN_SIZES}) == 14
    for s in QWEN_SIZES:
        # The pipeline rounds down to a multiple of 32: a real run asked for 1200x896 and produced 1184x896.
        assert s["width"] % 32 == 0 and s["height"] % 32 == 0, s
        want_w, want_h = map(int, s["aspect_ratio"].split(":"))
        assert abs(s["width"] / s["height"] - want_w / want_h) / (want_w / want_h) < 0.01, s  # the label tells the truth
    for s in native:
        twin = small[s["aspect_ratio"]]
        assert 0.9e6 <= twin["width"] * twin["height"] <= 1.1e6  # ~1 MP, the size that was measured to fit in 24 GiB
        assert s["width"] * s["height"] > 2.4e6 * 1.5
    assert (QWEN_SIZES[0]["width"], QWEN_SIZES[0]["height"]) == (1024, 1024)


def test_aspect_ratio_alone_gives_the_fast_tier_and_explicit_sizes_give_native():
    caps = QWEN.capabilities
    assert resolve_image_size(caps, {})["width"] == 1024
    assert (resolve_image_size(caps, {"aspect_ratio": "16:9"})["width"], resolve_image_size(caps, {"aspect_ratio": "16:9"})["height"]) == (1376, 768)
    native = resolve_image_size(caps, {"width": 2752, "height": 1536})
    assert native["tier"] == "native"
    with pytest.raises(ApiError):
        resolve_image_size(caps, {"width": 1025, "height": 1025})  # never rounded to a nearby size


def test_transparent_and_default_steps_are_part_of_the_accepted_params():
    accepted = validate_job(QWEN, "image.generate", "a dragon sticker", {"transparent": True, "aspect_ratio": "1:1"}, [])
    assert accepted["transparent"] is True and accepted["steps"] == 40 and accepted["width"] == 1024
    with pytest.raises(ApiError):
        validate_job(QWEN, "image.generate", "x", {"transparent": "yes"}, [])
    with pytest.raises(ApiError):
        validate_job(CATALOG["example-image"], "image.generate", "x", {"transparent": True}, [])  # that model has no alpha
    with pytest.raises(ApiError, match="mask"):
        validate_job(QWEN, "image.edit", "x", {}, [{"role": "source"}, {"role": "mask"}])  # not claimed until verified


# ---------------------------------------------------------------------------------------------- manifest


def test_manifest_marks_the_recommended_variant(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "detect_gpus", lambda ttl_s=30.0: [{"index": 0, "name": "RTX", "memory_total_mib": 24564}])
    monkeypatch.setattr(catalog, "detect_ram_gib", lambda: 32.0)
    cfg = Config(data_dir=tmp_path, models=[ModelConfig("qwen-image-2.1", "qwen_image", variant="int8")])
    entry = manifest_model(cfg, cfg.models[0], QWEN)
    assert entry["variant"] == "int8"
    assert entry["variants"]["int8"] == {
        **catalog.variant_public(QWEN.variants["int8"]), "selected": True, "recommended": False, "installed": False, "fit": "fits",
    }
    assert list(entry["variants"]) == FOUR and entry["variants"]["int8"]["disk_gib"] == 18
    assert [n for n, v in entry["variants"].items() if v["recommended"]] == ["int8-convrot"]  # best measured quality that fits
    assert not {"folder", "files", "runtime"} & set(entry["variants"]["int8"])  # internal fields never reach a client


def test_each_variant_has_its_own_weights_directory(tmp_path):
    cfg = Config(data_dir=tmp_path)
    a = weights_dir(cfg, ModelConfig("qwen-image-2.1", "qwen_image", variant="int8"))
    b = weights_dir(cfg, ModelConfig("qwen-image-2.1", "qwen_image", variant="other"))
    assert a != b and a.parent == b.parent  # the mechanism stays: a later model may offer a choice again


# ---------------------------------------------------------------------------------------------- install command


def _init(tmp_path) -> Path:
    assert main(["--data-dir", str(tmp_path), "init"]) == 0
    return tmp_path / "config.toml"


def test_install_refuses_without_the_license_key_and_prints_what_to_type(tmp_path, capsys):
    config = _init(tmp_path)
    before = config.read_text(encoding="utf-8")
    for extra in ([], ["--accept-license", "yes"]):
        assert main(["--data-dir", str(tmp_path), "models", "install", "qwen-image-2.1", *extra]) == 2
    out = capsys.readouterr().out
    assert "Qwen RESEARCH LICENSE" in out and "--accept-license qwen-research" in out and "not_allowed" in out
    assert config.read_text(encoding="utf-8") == before  # nothing was installed or written


def test_install_from_a_local_directory_records_the_choice(tmp_path):
    _init(tmp_path)
    weights = tmp_path / "downloaded"
    weights.mkdir()
    (weights / "model_index.json").write_text("{}", encoding="utf-8")
    argv = ["--data-dir", str(tmp_path), "models", "install", "qwen-image-2.1", "--variant", "int8",
            "--accept-license", "qwen-research", "--from-dir", str(weights), "--skip-env"]
    assert main(argv) == 0
    cfg = load(tmp_path)
    mc = cfg.model("qwen-image-2.1")
    assert mc is not None and mc.variant == "int8" and mc.enabled and Path(mc.weights_dir) == weights.resolve()
    record = json.loads((weights / "install.json").read_text(encoding="utf-8"))
    assert record["variant"] == "int8" and record["repo"] == "ixim/Image21-INT8" and len(record["revision"]) == 40
    assert main(argv) == 0 and len(load(tmp_path).models) == 1  # installing twice does not duplicate the entry


def test_install_rejects_a_directory_that_is_not_the_model(tmp_path, capsys):
    _init(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    code = main(["--data-dir", str(tmp_path), "models", "install", "qwen-image-2.1",
                 "--accept-license", "qwen-research", "--from-dir", str(empty), "--skip-env"])
    assert code == 2 and "model_index.json" in capsys.readouterr().out
    assert load(tmp_path).model("qwen-image-2.1") is None


def test_a_variant_that_is_not_offered_is_refused_and_the_choices_are_named(tmp_path, capsys):
    _init(tmp_path)
    capsys.readouterr()
    code = main(["--data-dir", str(tmp_path), "models", "install", "qwen-image-2.1", "--variant", "bf16", "--accept-license", "qwen-research"])
    out = capsys.readouterr().out
    assert code == 2 and "no variant 'bf16'" in out and "int8-convrot" in out and "mixed-p015" in out


def test_unknown_models_are_refused_and_minimax_asks_for_its_own_licence_key(tmp_path, capsys):
    _init(tmp_path)
    capsys.readouterr()  # drop init's output: it prints tmp_path, whose name contains this test's name
    assert main(["--data-dir", str(tmp_path), "models", "install", "minimax-h3-fl2va", "--accept-license", "x"]) == 2
    out = capsys.readouterr().out
    assert "--accept-license minimax-h3" in out and "MiniMax H3 Community License" in out
    assert main(["--data-dir", str(tmp_path), "models", "install", "no-such-model"]) == 2
    assert "unknown model" in capsys.readouterr().out
    assert main(["--data-dir", str(tmp_path), "models", "install", "example-image"]) == 2  # test-only models are not installable
    assert "unknown model" in capsys.readouterr().out
