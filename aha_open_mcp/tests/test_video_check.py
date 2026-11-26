"""Reading an mp4's own boxes and judging a video result on the FILE (plan 2026-11-24_minimax_h3_plan.md, M5).

Fixtures (tests/fixtures): ``h3_tiny.mp4`` is 12 frames of 96x64 at 24 fps with a stereo 32 kHz AAC track and the
comment "AI-generated (MiniMax H3)"; ``silent_tiny.mp4`` has the same picture and no sound.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from aha_open_mcp import jobs
from aha_open_mcp.mp4 import Mp4Error, probe_mp4

FIX = Path(__file__).parent / "fixtures"
GOOD = (FIX / "h3_tiny.mp4").read_bytes()
SILENT = (FIX / "silent_tiny.mp4").read_bytes()

CHECKS = ("size_matches", "fps_matches", "duration_matches", "audio_matches", "sha256_matches", "bytes_match")


def meta_of(data: bytes, **over) -> dict:
    """What a gateway result would say about the file, so the comparison starts from a true statement."""
    base = {"kind": "video", "width": 96, "height": 64, "fps": 24, "duration_s": 0.5, "has_audio": True,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return {**base, **over}


def test_it_reads_the_picture_the_frames_the_rate_the_sound_and_the_label():
    info = probe_mp4(GOOD)
    video, audio = info["video"], info["audio"]
    assert (video["width"], video["height"], video["samples"], video["fps"]) == (96, 64, 12, 24.0)
    assert video["duration_s"] == pytest.approx(0.5, abs=0.01)
    assert (audio["channels"], audio["sample_rate"]) == (2, 32000)
    assert info["comment"] == "AI-generated (MiniMax H3)" and info["tracks"] == 2


def test_a_file_without_sound_has_no_audio_track_and_no_label():
    info = probe_mp4(SILENT)
    assert info["audio"] is None and info["comment"] is None and info["tracks"] == 1
    assert (info["video"]["width"], info["video"]["height"], info["video"]["samples"]) == (96, 64, 12)


@pytest.mark.parametrize(
    "data",
    [GOOD[: len(GOOD) // 2], GOOD[:-9], b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, b"abc"],
    ids=["cut-in-the-middle", "cut-near-the-end", "a-png", "too-short"],
)
def test_a_file_that_is_not_a_whole_mp4_is_refused(data):
    with pytest.raises(Mp4Error):
        probe_mp4(data)


def test_a_file_with_pictures_but_no_index_is_refused():
    """What a writer that died before it wrote `moov` leaves: ftyp and mdat only."""
    pos, boxes = 0, {}
    while pos < len(GOOD):
        size, kind = struct.unpack(">I4s", GOOD[pos : pos + 8])
        boxes[kind] = GOOD[pos : pos + size]
        pos += size
    with pytest.raises(Mp4Error):
        probe_mp4(boxes[b"ftyp"] + boxes[b"mdat"])


def test_a_file_that_says_what_the_job_says_passes_every_comparison():
    r = jobs.judge_video(GOOD, meta_of(GOOD))
    assert r["fetched"] and r["parsed"]
    assert all(r[k] for k in CHECKS), r
    assert (r["width"], r["height"], r["frames"], r["audio_channels"], r["audio_sample_rate"], r["comment"]) == (
        96, 64, 12, 2, 32000, "AI-generated (MiniMax H3)")


@pytest.mark.parametrize(
    ("meta_over", "broken"),
    [
        ({"width": 97}, "size_matches"),
        ({"height": 65}, "size_matches"),
        ({"fps": 25}, "fps_matches"),
        ({"duration_s": 5.17}, "duration_matches"),
        ({"has_audio": False}, "audio_matches"),
        ({"sha256": "0" * 64}, "sha256_matches"),
        ({"bytes": 1}, "bytes_match"),
    ],
)
def test_each_disagreement_between_the_job_and_the_file_is_named(meta_over, broken):
    r = jobs.judge_video(GOOD, meta_of(GOOD, **meta_over))
    assert r[broken] is False
    assert all(r[k] for k in CHECKS if k != broken), r  # only the one that is wrong is wrong


def test_a_job_that_claims_sound_for_a_silent_file_is_caught():
    r = jobs.judge_video(SILENT, meta_of(SILENT, has_audio=True))
    assert r["has_audio_track"] is False and r["audio_matches"] is False


def test_a_file_whose_picture_size_was_changed_inside_is_caught_when_the_job_still_says_the_old_one():
    needle = struct.pack(">HH", 96, 64)
    assert GOOD.count(needle) >= 1
    patched = GOOD.replace(needle, struct.pack(">HH", 98, 64))
    r = jobs.judge_video(patched, meta_of(GOOD))
    assert r["parsed"] and r["size_matches"] is False


def test_a_cut_file_is_reported_as_not_parsed_not_as_an_exception():
    r = jobs.judge_video(GOOD[: len(GOOD) // 2], meta_of(GOOD))
    assert r["fetched"] is True and r["parsed"] is False and r["parse_error"]
