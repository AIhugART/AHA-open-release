"""What MiniMax H3 declares and accepts (plan 2026-11-24_minimax_h3_plan.md, M3): only what was run for real, every param complete."""

from __future__ import annotations

from pathlib import Path

import pytest

from aha_open import catalog
from aha_open.catalog import CATALOG, manifest_model, validate_job, video_num_frames
from aha_open.config import Config, ModelConfig
from aha_open.errors import ApiError

H3 = CATALOG["minimax-h3-fl2va"]


def ok(params=None, inputs=None):
    return validate_job(H3, "video.generate", "a lighthouse", params or {}, inputs or [], variant="mixed")


# ---------------------------------------------------------------------------------------------- what is accepted


def test_the_default_request_is_the_native_size_for_five_seconds_with_every_value_filled_in():
    accepted = ok()
    assert (accepted["width"], accepted["height"]) == (1344, 768)  # the model's default short edge
    assert accepted["num_frames"] == 124 and accepted["steps"] == 20
    assert accepted["aspect_ratio"] == "16:9" and accepted["short_edge"] == 768 and accepted["duration_s"] == 5


def test_the_preview_size_is_the_exact_size_that_was_measured():
    accepted = ok({"short_edge": 352})
    assert (accepted["width"], accepted["height"]) == (608, 352)


def test_every_declared_size_is_a_multiple_of_32_and_matches_its_short_edge():
    for size in H3.capabilities["sizes"]:
        assert size["width"] % 32 == 0 and size["height"] % 32 == 0
        assert min(size["width"], size["height"]) == size["short_edge"]
    assert {s["short_edge"] for s in H3.capabilities["sizes"]} == set(H3.capabilities["short_edges"])


@pytest.mark.parametrize(
    ("params", "needle"),
    [
        ({"duration_s": 10}, "duration_s must be between 5 and 5"),  # the README says 4-15 s, but only 5 s was run
        ({"duration_s": 4}, "duration_s must be between"),
        ({"aspect_ratio": "9:16"}, "not supported"),  # only 16:9 was run
        ({"aspect_ratio": "1:1"}, "not supported"),
        ({"short_edge": 480}, "short_edge must be one of"),
        ({"steps": 8}, "steps is outside"),
        ({"steps": 50}, "steps is outside"),
        ({"width": 640, "height": 352}, "unknown params"),  # a size is chosen by short_edge, never given freely
    ],
)
def test_what_was_not_measured_is_refused_before_a_job_exists(params, needle):
    with pytest.raises(ApiError) as err:
        ok(params)
    assert err.value.status == 400 and needle in err.value.message


@pytest.mark.parametrize("role", ["first_frame", "last_frame"])
def test_a_frame_input_is_refused_with_400_instead_of_being_accepted_and_failed(role):
    with pytest.raises(ApiError) as err:
        ok(inputs=[{"role": role}])
    assert err.value.status == 400 and "does not accept a" in err.value.message


def test_a_prompt_is_required():
    with pytest.raises(ApiError):
        validate_job(H3, "video.generate", " ", {}, [], variant="mixed")


# ---------------------------------------------------------------------------------------------- the frame grid


@pytest.mark.parametrize(
    ("seconds", "fps", "grid", "frames"),
    [
        (5, 24, {"step": 17, "base": 5}, 124),  # 120 -> the next 17n+5
        (0.1, 24, {"step": 17, "base": 5}, 5),  # never below the grid's base
        (0.5, 24, {"step": 17, "base": 5}, 22),  # 12 frames -> the next 17n+5
        (2, 8, {"step": 4, "base": 1}, 17),
        (3, 8, None, 24),  # no grid: just round(duration * fps)
    ],
)
def test_frames_snap_up_to_the_models_grid(seconds, fps, grid, frames):
    assert video_num_frames(seconds, fps, grid) == frames
    if grid:
        assert (frames - grid["base"]) % grid["step"] == 0


# ---------------------------------------------------------------------------------------------- what the manifest says


def test_the_manifest_declares_the_measured_variant_and_size_tiers(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "detect_gpus", lambda ttl_s=30.0: [{"index": 0, "name": "RTX", "memory_total_mib": 24564}])
    monkeypatch.setattr(catalog, "detect_ram_gib", lambda: 32.0)
    cfg = Config(data_dir=tmp_path, models=[ModelConfig(id="minimax-h3-fl2va", worker="minimax_h3", variant="mixed")])
    entry = manifest_model(cfg, cfg.models[0], H3)
    assert entry["tasks"] == ["video.generate"] and entry["license"]["required_ui_label"] == "MiniMax H3"
    assert entry["license"]["excluded_territories"] == ["EU", "GB", "KR", "US"]
    variant = entry["variants"]["mixed"]
    assert variant["selected"] is True and variant["recommended"] is True and variant["disk_gib"] == 46
    assert variant["vram_gib"] == {"preview": 16.7, "native": 19.6}
    caps = entry["capabilities"]
    assert caps["first_frame"] is False and caps["last_frame"] is False and caps["audio"] == {"channels": 2, "sample_rate": 32000}


def test_the_vram_rule_offers_h3_from_16_gb_cards_with_the_build_that_runs_there(tmp_path, monkeypatch):
    from aha_open.hardware import recommend_variant_for

    assert H3.min_vram_mib == {"int8": 23552, "mixed": 18432, "nvfp4": 15872}  # docs/05_models.md section 2
    for gpu_mib, expected in ((8192, None), (12288, None), (16376, "nvfp4"), (20480, "mixed"), (24564, "mixed")):
        assert recommend_variant_for("minimax-h3-fl2va", gpu_mib, 32.0)[0] == expected, gpu_mib
    reason = recommend_variant_for("minimax-h3-fl2va", 12288, 32.0)[1]
    assert "at least 16 GiB" in reason
    for name, info in H3.variants.items():  # a build is offered with a size only if that size was run
        assert info["vram_gib"]["preview"] is not None, name


@pytest.mark.parametrize("gpu_mib,variant,needle", [
    (12288, "mixed", "needs a GPU of at least 16 GB; this server has 12 GB"),
    (16376, "mixed", "variants that fit here: nvfp4"),
])
def test_a_card_that_is_too_small_is_told_what_would_work(tmp_path, monkeypatch, gpu_mib, variant, needle):
    monkeypatch.setattr(catalog, "detect_gpus", lambda ttl_s=30.0: [{"index": 0, "name": "RTX", "memory_total_mib": gpu_mib}])
    monkeypatch.setattr(catalog, "variant_installed", lambda *a: True)
    cfg = Config(data_dir=tmp_path, models=[ModelConfig(id="minimax-h3-fl2va", worker="minimax_h3", variant=variant, weights_dir=str(tmp_path))])
    (tmp_path / "model_index.json").write_text("{}")
    for rel in H3.variants[variant]["files"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("{}")
    monkeypatch.setattr(catalog, "worker_python", lambda *a: Path("python"))
    state, detail = catalog.model_state(cfg, cfg.models[0], H3)
    assert state == "unsupported_hardware" and needle in detail, (state, detail)


def test_a_configuration_left_over_from_the_bf16_days_says_what_to_do(tmp_path):
    cfg = Config(data_dir=tmp_path, models=[ModelConfig(id="minimax-h3-fl2va", worker="minimax_h3", variant="bf16")])
    state, detail = catalog.model_state(cfg, cfg.models[0], H3)
    assert state == "error" and "not offered any more" in detail and "mixed" in detail and "models install minimax-h3-fl2va" in detail
