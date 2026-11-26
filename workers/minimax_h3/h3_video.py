"""Write and re-read the mp4 the MiniMax H3 worker returns.

* H.264 through the OpenH264 encoder (BSD-2-Clause, bundled in the PyAV wheel). The same wheel also carries x264/x265, which are
  GPL upstream; this worker never selects them (THIRD_PARTY_LICENSES.md, MiniMax H3 worker).
* AAC stereo at the model's 32 kHz.
* The licence asks that machine-made content be disclosed (Exhibit A.12): the container's ``comment`` says so. The prompt is
  never written into the file.
* ``probe`` reads the file back so the worker reports what is in the file, not what it meant to write.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

AI_LABEL = "AI-generated (MiniMax H3)"
VIDEO_CODEC = "libopenh264"
BITS_PER_PIXEL = 0.25  # ~12 Mbit/s at 1344x768x24: little visible loss on generated footage, files of a few MB per 5 s


def write_mp4(path: str | Path, frames: np.ndarray, audio: np.ndarray, sampling_rate: int, fps: int = 24) -> None:
    """``frames``: uint8 [T, H, W, 3]; ``audio``: float [channels, samples] in [-1, 1]."""
    import av

    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be uint8 [T, H, W, 3], got {frames.dtype} {frames.shape}")
    count, height, width, _ = frames.shape
    audio = np.ascontiguousarray(np.clip(audio, -1.0, 1.0).astype(np.float32))
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"audio must be [2, samples], got {audio.shape}")
    with av.open(str(path), mode="w", format="mp4") as container:
        container.metadata["comment"] = AI_LABEL
        video = container.add_stream(VIDEO_CODEC, rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.bit_rate = int(width * height * fps * BITS_PER_PIXEL)
        sound = container.add_stream("aac", rate=sampling_rate)
        sound.layout = "stereo"
        for frame in frames:
            for packet in video.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        step = 1024
        for start in range(0, audio.shape[1], step):
            chunk = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio[:, start : start + step]), format="fltp", layout="stereo")
            chunk.sample_rate, chunk.pts = sampling_rate, start
            for packet in sound.encode(chunk):
                container.mux(packet)
        for packet in sound.encode():
            container.mux(packet)


def probe(path: str | Path) -> dict:
    """What the file holds: decoded frame count, size, frame rate, duration, audio stream, the container comment."""
    import av

    with av.open(str(path)) as container:
        video = container.streams.video[0]
        width, height = video.codec_context.width, video.codec_context.height
        fps = float(video.average_rate) if video.average_rate else 0.0
        audio = container.streams.audio[0] if container.streams.audio else None
        info = {
            "width": width,
            "height": height,
            "fps": fps,
            "codec": video.codec_context.name,
            "audio_rate": audio.codec_context.sample_rate if audio else None,
            "audio_channels": audio.codec_context.channels if audio else 0,
            "comment": container.metadata.get("comment", ""),
        }
        info["frames"] = sum(1 for _ in container.decode(video=0))
    info["duration_s"] = round(info["frames"] / fps, 3) if fps else 0.0
    return info
