"""What this machine can run (plan 2026-11-25_minimax_h3_variants_plan.md, V1): GPU generation, the operator's memory limit, fit per size."""

from __future__ import annotations

import subprocess
import tomllib

import pytest

from aha_open import catalog, config, hardware
from aha_open.catalog import QWEN_VARIANTS
from aha_open.hardware import fit_by_tier, fit_for, gpu_budget_mib, recommend_variant_for

# what the code answered before this change, GPU MiB -> (fit of int8, int8-convrot, mixed-p010, mixed-p015), suggestion
QWEN_BEFORE = {
    None: (("too_small",) * 4, None),
    8192: (("too_small",) * 4, None),
    9216: (("too_small", "tight", "tight", "tight"), "int8-convrot"),
    10240: (("tight", "fits", "fits", "fits"), "int8-convrot"),
    11264: (("fits",) * 4, "int8-convrot"),
    24564: (("fits",) * 4, "int8-convrot"),
}


@pytest.mark.parametrize("gpu_mib", list(QWEN_BEFORE))
def test_qwen_answers_did_not_change(gpu_mib):
    fits, suggestion = QWEN_BEFORE[gpu_mib]
    assert tuple(fit_for(v, gpu_mib) for v in QWEN_VARIANTS.values()) == fits
    assert recommend_variant_for("qwen-image-2.1", gpu_mib, 32.0)[0] == suggestion


VIDEO = {"vram_gib": {"preview": 14.0, "native": 19.6}}


def test_a_video_variant_is_judged_by_its_smallest_measured_size_and_per_size():
    assert fit_for(VIDEO, 16384) == "fits"  # the small size runs
    assert fit_by_tier(VIDEO, 16384) == {"preview": "fits", "native": "too_small"}
    assert fit_by_tier(VIDEO, 20480) == {"preview": "fits", "native": "tight"}
    assert fit_by_tier(VIDEO, 24564) == {"preview": "fits", "native": "fits"}
    assert fit_for(VIDEO, 12288) == "too_small"


def test_an_image_models_offloaded_peaks_never_hide_a_size():
    # Qwen int8 native peaked at 26.9 GiB with offload on and still runs on 24 GiB: that is not a minimum
    assert fit_by_tier(QWEN_VARIANTS["int8"], 24564) == {}


def test_a_size_that_was_not_measured_never_refuses():
    info = {"vram_gib": {"preview": 17.1, "native": None}}
    assert fit_by_tier(info, 8192) == {"preview": "too_small", "native": "unknown"}
    assert fit_for({"vram_gib": {"preview": None}}, 4096) == "unknown"


def test_the_budget_is_the_smaller_of_the_card_and_the_operators_limit(tmp_path):
    gpu = {"index": 0, "memory_total_mib": 24564}
    assert gpu_budget_mib(config.Config(data_dir=tmp_path), gpu) == 24564
    assert gpu_budget_mib(config.Config(data_dir=tmp_path, gpu_memory_limit_gib=15.5), gpu) == 15872
    assert gpu_budget_mib(config.Config(data_dir=tmp_path, gpu_memory_limit_gib=48.0), gpu) == 24564
    assert gpu_budget_mib(config.Config(data_dir=tmp_path, gpu_memory_limit_gib=15.5), None) is None


def test_the_memory_limit_round_trips_through_the_config_file(tmp_path):
    cfg = config.Config(data_dir=tmp_path, gpu_memory_limit_gib=15.5)
    text = config.render(cfg)
    assert "[gpu]" in text
    assert config.from_dict(tomllib.loads(text), tmp_path).gpu_memory_limit_gib == 15.5
    assert "[gpu]" not in config.render(config.Config(data_dir=tmp_path))


@pytest.mark.parametrize("bad", [-1, 0.5, "16", True])
def test_a_limit_that_makes_no_sense_is_refused(tmp_path, bad):
    with pytest.raises(config.ConfigError, match="memory_limit_gib"):
        config.from_dict({"gpu": {"memory_limit_gib": bad}}, tmp_path)


class _Result:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout, self.returncode = stdout, returncode


def _fake_smi(monkeypatch, capability: str | None):
    monkeypatch.setattr(catalog.shutil, "which", lambda name: "nvidia-smi")

    def run(args, **kwargs):
        if "compute_cap" in args[1]:
            if capability is None:
                return _Result('Field "compute_cap" is not a valid field to query.\n', 2)
            return _Result(f"0, {capability}\n")
        return _Result("0, NVIDIA GeForce RTX 3090 Ti, 24564\n")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(catalog, "_gpu_cache", (0.0, []))


def test_gpu_detection_reports_the_compute_capability(monkeypatch):
    _fake_smi(monkeypatch, "8.6")
    assert catalog.detect_gpus(ttl_s=0)[0]["compute_cap"] == "8.6"


def test_a_driver_without_compute_cap_still_finds_the_gpu(monkeypatch):
    _fake_smi(monkeypatch, None)
    gpus = catalog.detect_gpus(ttl_s=0)
    assert gpus == [{"index": 0, "name": "NVIDIA GeForce RTX 3090 Ti", "memory_total_mib": 24564, "compute_cap": None}]


def test_hardware_does_not_import_torch():
    assert "torch" not in hardware.__dict__


def test_a_size_that_ran_out_of_memory_is_too_small_on_that_card_and_unknown_above():
    info = {"vram_gib": {"preview": 21.4, "native": None}, "too_small_up_to_gib": {"native": 24}}
    assert fit_by_tier(info, 24564) == {"preview": "fits", "native": "too_small"}
    assert fit_by_tier(info, 32 * 1024) == {"preview": "fits", "native": "unknown"}


def test_a_run_that_failed_on_a_card_outweighs_a_peak_that_would_have_fitted_it():
    # Ref2VA NVFP4: peak 14.55 GiB, yet out of memory under a 15.5 GiB cap
    info = {"vram_gib": {"preview": 14.6}, "too_small_up_to_gib": {"preview": 16}}
    assert fit_by_tier(info, 16376) == {"preview": "too_small"} and fit_for(info, 16376) == "too_small"
    assert fit_for(info, 24564) == "fits"


def test_the_reference_model_is_not_offered_below_the_card_it_ran_on():
    assert recommend_variant_for("minimax-h3-ref2va", 16376, 32.0)[0] is None
    assert recommend_variant_for("minimax-h3-ref2va", 24564, 32.0)[0] == "mixed"
