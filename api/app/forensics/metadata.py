"""Metadata / container forensic branch.

`inspect_file(path)` reads container/codec/sample-rate/tags via `soundfile` first
(fast, exact for PCM WAV/FLAC/OGG) and falls back to parsing `ffmpeg -i <path>`
stderr (via the imageio-ffmpeg bundled binary, or $FFMPEG_PATH if it resolves to an
existing executable) for formats soundfile can't open or to recover encoder tags.
No ffprobe dependency required -- `ffmpeg -i` prints the same stream/format info on
stderr and exits non-zero without an output, which is fine, we only parse stderr.

`analyze_metadata(info)` turns that raw info into an evidence dict. Metadata is
circumstantial: it is trivially forged or stripped, so summaries always say so.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any

from . import _common as C

_LOSSY_CODEC_HINTS = (
    "mp3", "mp2", "aac", "opus", "vorbis", "ac3", "eac3", "wma", "amr",
)
_TTS_ENCODER_HINTS = (
    "elevenlabs", "tts", "coqui", "tortoise", "bark", "vits", "tacotron",
    "waveglow", "hifigan", "diffwave", "wavegrad", "styletts", "xtts",
)


def _resolve_ffmpeg() -> str | None:
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    which = shutil.which("ffmpeg")
    return which


def _parse_ffmpeg_stderr(text: str) -> dict[str, Any]:
    info: dict[str, Any] = {"metadata": {}}

    m = re.search(r"Input #0,\s*([^,]+),\s*from", text)
    if m:
        info["container"] = m.group(1).strip()

    dur_m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    if dur_m:
        h, mnt, s = dur_m.groups()
        info["duration_s"] = int(h) * 3600 + int(mnt) * 60 + float(s)

    bitrate_m = re.search(r"bitrate:\s*(\d+)\s*kb/s", text)
    if bitrate_m:
        info["bitrate"] = int(bitrate_m.group(1)) * 1000

    stream_m = re.search(
        r"Audio:\s*([\w.]+).*?,\s*(\d+)\s*Hz,\s*([\w.() ]+?),\s*([\w0-9]+)",
        text,
    )
    if stream_m:
        info["codec"] = stream_m.group(1).strip()
        info["sample_rate"] = int(stream_m.group(2))
        layout = stream_m.group(3).strip()
        info["channel_layout"] = layout
        if "mono" in layout:
            info["channels"] = 1
        elif "stereo" in layout:
            info["channels"] = 2
        else:
            ch_m = re.match(r"(\d+)", layout)
            info["channels"] = int(ch_m.group(1)) if ch_m else None
        sample_fmt = stream_m.group(4)
        bd_m = re.search(r"(\d+)", sample_fmt)
        if bd_m and ("s" in sample_fmt or "u" in sample_fmt or "flt" in sample_fmt):
            info["bit_depth"] = int(bd_m.group(1))

    for tag_match in re.finditer(r"^\s{4,6}([\w.\- ]+?)\s*:\s*(.+)$", text, re.MULTILINE):
        key, val = tag_match.group(1).strip(), tag_match.group(2).strip()
        if key.lower() in ("duration", "bitrate", "stream", "start"):
            continue
        info["metadata"][key] = val

    return info


def inspect_file(path: str) -> dict[str, Any]:
    """Return raw container/codec/tag info for `path`. Best-effort; missing fields are None."""
    info: dict[str, Any] = {
        "container": None, "codec": None, "sample_rate": None, "channels": None,
        "bit_depth": None, "duration_s": None, "bitrate": None, "metadata": {},
    }

    try:
        import soundfile as sf

        sfinfo = sf.info(path)
        info["container"] = (sfinfo.format or "").lower() or None
        info["codec"] = (sfinfo.subtype or "").lower() or None
        info["sample_rate"] = int(sfinfo.samplerate)
        info["channels"] = int(sfinfo.channels)
        info["duration_s"] = float(sfinfo.frames) / sfinfo.samplerate if sfinfo.samplerate else None
        subtype_bits = re.search(r"(\d+)", sfinfo.subtype or "")
        if subtype_bits:
            info["bit_depth"] = int(subtype_bits.group(1))
    except Exception:
        pass

    ffmpeg = _resolve_ffmpeg()
    if ffmpeg and (info["codec"] is None or not info["metadata"]):
        try:
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", path],
                capture_output=True, text=True, timeout=15,
            )
            ff_info = _parse_ffmpeg_stderr(proc.stderr or "")
            for k, v in ff_info.items():
                if k == "metadata":
                    info["metadata"].update(v)
                elif info.get(k) in (None, "") and v is not None:
                    info[k] = v
        except Exception:
            pass

    return info


def analyze_metadata(info: dict[str, Any]) -> dict:
    """Evidence dict from `inspect_file` output. Metadata is evidence, never proof."""
    flags: list[str] = []
    suspicion_terms: list[float] = []

    codec = (info.get("codec") or "").lower()
    container = (info.get("container") or "").lower()
    duration_s = info.get("duration_s")
    tags = info.get("metadata") or {}

    is_lossy = any(hint in codec or hint in container for hint in _LOSSY_CODEC_HINTS)
    if is_lossy:
        flags.append(f"lossy codec/container detected ({codec or container})")
        suspicion_terms.append(0.3)

    if duration_s is not None and duration_s <= 0:
        flags.append("zero or invalid duration")
        suspicion_terms.append(0.2)

    sample_rate = info.get("sample_rate")
    if sample_rate is not None and sample_rate not in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
        flags.append(f"unusual sample rate ({sample_rate} Hz), possible resampling")
        suspicion_terms.append(0.2)

    tag_blob = " ".join(f"{k}:{v}" for k, v in tags.items()).lower()
    for hint in _TTS_ENCODER_HINTS:
        if hint in tag_blob:
            flags.append(f"encoder/tag mentions '{hint}' (possible TTS tool signature)")
            suspicion_terms.append(0.5)
            break

    encoder = str(tags.get("encoder", "")).lower()
    if "lavf" in encoder or "ffmpeg" in encoder:
        flags.append(f"re-encoded with a generic transcoding tool ({tags.get('encoder')})")
        suspicion_terms.append(0.15)

    suspicion = float(min(sum(suspicion_terms), 1.0)) if suspicion_terms else 0.0
    if not flags:
        summary = "No unusual metadata found. Metadata is easily stripped or forged and is evidence only, never proof."
    else:
        summary = (
            "Metadata anomalies: " + "; ".join(flags) + ". "
            "Metadata is circumstantial evidence only -- it can be trivially edited or stripped."
        )

    features = {
        "sample_rate_hz": C.safe_float(sample_rate),
        "duration_s": C.safe_float(duration_s),
        "bit_depth": C.safe_float(info.get("bit_depth")),
        "channels": C.safe_float(info.get("channels")),
        "bitrate": C.safe_float(info.get("bitrate")),
        "is_lossy_codec": 1.0 if is_lossy else 0.0,
        "num_metadata_tags": float(len(tags)),
    }

    return {
        "summary": summary,
        "suspicion": suspicion if flags else None,
        "features": features,
        "flags": flags,
        "used_in_score": False,
    }


def analyze(audio, sr, **ctx) -> dict:
    """Convenience wrapper matching the branch `analyze(audio, sr, **ctx)` signature.

    Metadata isn't derivable from raw samples alone; pass `path=<file path>` in ctx
    to get real results, otherwise this returns an explicit not-available result.
    """
    path = ctx.get("path")
    if not path:
        return C.empty_result(
            "No file path provided; metadata analysis requires the original file.",
            ["no_path_provided"],
        )
    info = inspect_file(path)
    return analyze_metadata(info)


def extract_features(audio, sr) -> dict[str, float]:
    """Metadata features are not derivable from raw samples; returns zeros.

    Present for API consistency with the other branches / `features.py`.
    """
    return {
        "sample_rate_hz": 0.0, "duration_s": 0.0, "bit_depth": 0.0,
        "channels": 0.0, "bitrate": 0.0, "is_lossy_codec": 0.0, "num_metadata_tags": 0.0,
    }
