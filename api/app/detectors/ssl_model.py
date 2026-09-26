"""SSL front-end (wav2vec2 / XLS-R / WavLM) + learned layer weighting + attentive stats pooling -> 1 logit.

Shared by training (ml/train.py) and serving (api/app/detectors/primary.py) so there is exactly one model definition.
Output logit > 0 means "synthetic" (p_synth = sigmoid(logit)).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveStatsPool(nn.Module):
    def __init__(self, d: int, h: int = 128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x, mask):  # x [B,T,D], mask [B,T] bool
        a = self.att(x).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(a.float(), dim=1).to(x.dtype).unsqueeze(-1)
        mu = (w * x).sum(1)
        sd = ((w * (x - mu.unsqueeze(1)) ** 2).sum(1)).clamp_min(1e-6).sqrt()
        return torch.cat([mu, sd], -1)


class SSLDetector(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True, cache_dir: str | None = None, config=None,
                 num_layers: int | None = None):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        kw = {"num_hidden_layers": num_layers} if num_layers else {}
        if pretrained:
            # truncating num_hidden_layers keeps the first N pretrained transformer blocks
            self.ssl = AutoModel.from_pretrained(backbone, cache_dir=cache_dir, **kw)
        else:
            cfg = config if config is not None else AutoConfig.from_pretrained(backbone, cache_dir=cache_dir, **kw)
            self.ssl = AutoModel.from_config(cfg)
        # SpecAugment-style masking inside the backbone is disabled; we augment waveforms instead.
        for k in ("mask_time_prob", "mask_feature_prob", "layerdrop"):
            if hasattr(self.ssl.config, k):
                setattr(self.ssl.config, k, 0.0)
        self.backbone_name = backbone
        n_layers = self.ssl.config.num_hidden_layers + 1
        d = self.ssl.config.hidden_size
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU())
        self.pool = AttentiveStatsPool(256)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))

    def frame_mask(self, lengths: torch.Tensor, T: int) -> torch.Tensor:
        if hasattr(self.ssl, "_get_feat_extract_output_lengths"):
            fl = self.ssl._get_feat_extract_output_lengths(lengths)
        else:
            fl = torch.div(lengths, 320, rounding_mode="floor")
        fl = torch.clamp(fl, min=1, max=T)
        return torch.arange(T, device=lengths.device)[None] < fl[:, None]

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor | None = None, return_frames: bool = False):
        """wav: [B, N] float, per-utterance zero-mean/unit-var normalised; lengths: valid samples per row."""
        if lengths is None:
            lengths = torch.full((wav.shape[0],), wav.shape[1], device=wav.device, dtype=torch.long)
        att = (torch.arange(wav.shape[1], device=wav.device)[None] < lengths[:, None]).long()
        use_mask = getattr(self.ssl.config, "feat_extract_norm", "layer") == "layer"
        out = self.ssl(wav, attention_mask=att if use_mask else None, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)  # [L,B,T,D]
        w = torch.softmax(self.layer_w, 0).to(hs.dtype)
        x = (w[:, None, None, None] * hs).sum(0)
        x = self.proj(x)
        mask = self.frame_mask(lengths, x.shape[1])
        logit = self.head(self.pool(x, mask)).squeeze(-1)
        if return_frames:
            return logit, x, mask
        return logit


def normalize(wav: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Zero-mean / unit-variance over the valid region of each row; padding stays 0."""
    m = (torch.arange(wav.shape[1], device=wav.device)[None] < lengths[:, None]).to(wav.dtype)
    n = m.sum(1, keepdim=True).clamp_min(1)
    mu = (wav * m).sum(1, keepdim=True) / n
    var = (((wav - mu) * m) ** 2).sum(1, keepdim=True) / n
    return (wav - mu) / (var.sqrt() + 1e-7) * m


def save_detector(model: SSLDetector, out_dir: str | Path, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.ssl.config.save_pretrained(out_dir / "backbone_config")
    sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save(sd, out_dir / "model.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta | {"backbone": model.backbone_name}, indent=2))


def load_detector(model_dir: str | Path, device: str | torch.device = "cpu") -> tuple[SSLDetector, dict]:
    """Offline load (no network): builds the backbone from the saved config, then loads all weights."""
    from transformers import AutoConfig

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    cfg = AutoConfig.from_pretrained(model_dir / "backbone_config")
    m = SSLDetector(meta["backbone"], pretrained=False, config=cfg)
    sd = torch.load(model_dir / "model.pt", map_location="cpu")
    own = m.state_dict().keys()
    # masked_spec_embed is a training-only SpecAugment parameter (we disable masking); tolerate only that key.
    sd = {k: v for k, v in sd.items() if k in own or not k.endswith("masked_spec_embed")}
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return m.to(device).eval(), meta
