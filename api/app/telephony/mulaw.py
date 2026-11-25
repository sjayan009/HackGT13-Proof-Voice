"""G.711 μ-law codec in numpy (audioop is gone in Python 3.13+). Twilio Media Streams carry 8 kHz μ-law."""
from __future__ import annotations

import numpy as np

_BIAS, _CLIP = 0x84, 32635


def _decode_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    mag = (((u & 0x0F) << 3) + _BIAS << ((u >> 4) & 7)) - _BIAS
    return (np.where(u & 0x80, -mag, mag) / 32768.0).astype(np.float32)


_DECODE = _decode_table()


def decode(b: bytes) -> np.ndarray:
    """μ-law bytes -> float32 in [-1, 1]."""
    return _DECODE[np.frombuffer(b, np.uint8)]


def encode(x: np.ndarray) -> bytes:
    """float32 in [-1, 1] -> μ-law bytes."""
    s = np.clip(np.round(np.asarray(x, np.float64) * 32768), -32768, 32767).astype(np.int32) >> 2  # 14-bit, as G.711
    neg = s < 0
    m = np.minimum(np.where(neg, -s, s), _CLIP >> 2) + (_BIAS >> 2)
    exp = np.clip(np.floor(np.log2(m)).astype(np.int32) - 5, 0, 7)
    mant = (m >> (exp + 1)) & 0x0F
    return (((exp << 4) | mant) ^ np.where(neg, 0x7F, 0xFF)).astype(np.uint8).tobytes()


if __name__ == "__main__":
    x = np.sin(np.linspace(0, 200, 8000)).astype(np.float32) * 0.5
    y = decode(encode(x))
    assert np.max(np.abs(x - y)) < 0.02, np.max(np.abs(x - y))
    assert decode(encode(np.zeros(4))).max() == 0.0
    try:  # bit-exact against the reference implementation where it still exists
        import audioop  # noqa: PLC0415

        s = np.arange(-32768, 32768)
        assert audioop.lin2ulaw(s.astype("<i2").tobytes(), 2) == encode(s / 32768), "encode mismatch vs audioop"
        ref = np.frombuffer(audioop.ulaw2lin(encode(x), 2), "<i2") / 32768.0
        assert np.allclose(ref, y), "decode mismatch vs audioop"
        print("bit-exact vs audioop")
    except ImportError:
        pass
    print("mulaw ok")
