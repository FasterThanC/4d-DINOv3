"""L-TAE temporal model + small decoder that turns saved token features into a crop map.

L-TAE follows Sainte Fare Garnot & Landrieu (2020) and the official implementation in
github.com/VSainteuf/utae-paps (MIT licence): a shared temporal attention per token,
with learned master queries per head and sinusoidal encoding of the acquisition day.

This is a self-contained copy so Pair 2 does not have to wait for Pair 1. When Pair 1's
temporal model and decoder are final, use theirs for the final numbers of both experiments.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=1):
        super().__init__()
        self.repeat = repeat
        denom = torch.pow(T, 2 * (torch.arange(d).float() // 2) / d)
        self.register_buffer("denom", denom, persistent=False)

    def forward(self, days):                                  # (N, T) -> (N, T, d * repeat)
        x = days[..., None].float() / self.denom
        out = torch.empty_like(x)
        out[..., 0::2] = torch.sin(x[..., 0::2])
        out[..., 1::2] = torch.cos(x[..., 1::2])
        return out.repeat(1, 1, self.repeat)


PACK = 10000.0   # date_mode "both": position and day arrive packed as pos * PACK + day


class BothEncoder(nn.Module):
    """Time input "both" (Experiment 1 design): in every head, half of the dimensions encode
    the position of the image, half its acquisition day. Input: pos * PACK + day (exact in
    float32 while pos * PACK < 2**24, i.e. fewer than ~160 images at pos_scale 10)."""

    def __init__(self, d_head, T=1000, repeat=1):
        super().__init__()
        self.pos = PositionalEncoder(d_head // 2, T=T)
        self.day = PositionalEncoder(d_head - d_head // 2, T=T)
        self.repeat = repeat

    def forward(self, packed):
        pos = torch.div(packed, PACK, rounding_mode="floor")
        day = packed - pos * PACK
        one = torch.cat([self.pos(pos), self.day(day)], dim=-1)
        return one.repeat(1, 1, self.repeat)


def masked_group_norm(v, gn, valid=None):
    """GroupNorm over (channels in group x dates) that ignores padded dates.

    v: (n, T, C); gn: nn.GroupNorm (for its groups, eps, weight, bias); valid: (n, T) bool.
    Without padding this equals gn(v.transpose(1, 2)).transpose(1, 2), which is what the
    official L-TAE does. The official version also counts the zero-padded dates, so a
    patch's output depends on how long the other series in its batch are; this one does not.
    """
    n, T, C = v.shape
    G = gn.num_groups
    x = v.reshape(n, T, G, C // G).float()
    w = torch.ones(n, T, device=v.device) if valid is None else valid.float()
    w = w[:, :, None, None]
    cnt = w.sum(1, keepdim=True).clamp(min=1) * (C // G)
    mean = (x * w).sum(dim=(1, 3), keepdim=True) / cnt
    var = (((x - mean) ** 2) * w).sum(dim=(1, 3), keepdim=True) / cnt
    x = (x - mean) / torch.sqrt(var + gn.eps)
    x = x.reshape(n, T, C)
    if gn.affine:
        x = x * gn.weight + gn.bias
    return x.to(v.dtype)


class LTAE(nn.Module):
    """Input (B, T, C, h, w) -> output (B, mlp[-1], h, w). pad_mask (B, T) is True for padded dates."""

    def __init__(self, in_channels, d_model=256, n_head=16, d_k=4, mlp=(256, 128),
                 dropout=0.2, T=1000, use_dates=True, return_att=False, time_enc="date"):
        super().__init__()
        assert mlp[0] == d_model and d_model % n_head == 0 and in_channels % n_head == 0
        self.n_head, self.d_k, self.return_att = n_head, d_k, return_att
        self.in_norm = nn.GroupNorm(n_head, in_channels)
        self.inconv = nn.Linear(in_channels, d_model)
        enc = BothEncoder if time_enc == "both" else PositionalEncoder
        self.pos = enc(d_model // n_head, T=T, repeat=n_head) if use_dates else None
        self.Q = nn.Parameter(torch.randn(n_head, d_k) * math.sqrt(2.0 / d_k))
        self.fc_k = nn.Linear(d_model, n_head * d_k)
        nn.init.normal_(self.fc_k.weight, mean=0, std=math.sqrt(2.0 / d_k))
        layers = []
        for a, b in zip(mlp[:-1], mlp[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.GroupNorm(n_head, mlp[-1])

    def forward(self, x, days, pad_mask=None):
        B, T, C, h, w = x.shape
        n = B * h * w
        v = x.permute(0, 3, 4, 1, 2).reshape(n, T, C)
        pm = None if pad_mask is None else pad_mask[:, None, None, :].expand(B, h, w, T).reshape(n, T)
        v = masked_group_norm(v, self.in_norm, None if pm is None else ~pm)
        v = self.inconv(v)                                                  # n, T, d_model
        if self.pos is not None:
            d = days[:, None, None, :].expand(B, h, w, T).reshape(n, T)
            v = v + self.pos(d)
        k = self.fc_k(v).view(n, T, self.n_head, self.d_k).permute(2, 0, 1, 3)   # H, n, T, dk
        att = torch.einsum("hd,hntd->hnt", self.Q, k) / math.sqrt(self.d_k)
        if pm is not None:
            att = att.masked_fill(pm[None], -1e4)
        att = att.softmax(dim=-1)                                           # H, n, T
        vh = v.view(n, T, self.n_head, -1).permute(2, 0, 1, 3)             # H, n, T, d/H
        out = torch.einsum("hnt,hntd->hnd", att, vh)                        # H, n, d/H
        out = out.permute(1, 0, 2).reshape(n, -1)                           # n, d_model
        out = self.out_norm(self.dropout(self.mlp(out)))
        out = out.view(B, h, w, -1).permute(0, 3, 1, 2)
        if self.return_att:
            return out, att.view(self.n_head, B, h, w, T)
        return out


class MeanPool(nn.Module):
    """Average over valid dates (baseline: no dates, no attention)."""

    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x, days, pad_mask=None):
        if pad_mask is None:
            m = x.mean(1)
        else:
            valid = (~pad_mask).float()[:, :, None, None, None]
            m = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.proj(m)


def conv_block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())


class Decoder(nn.Module):
    """(B, C, 16, 16) -> (B, num_classes, 128, 128): three x2 upsampling stages."""

    def __init__(self, in_channels=128, num_classes=19, out_size=128):
        super().__init__()
        self.out_size = out_size
        self.blocks = nn.ModuleList([conv_block(in_channels, 128), conv_block(128, 64),
                                     conv_block(64, 64), conv_block(64, 32)])
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x = self.blocks[0](x)
        for blk in self.blocks[1:]:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return self.head(x)


class SegModel(nn.Module):
    def __init__(self, in_channels, temporal="ltae", num_classes=19, use_dates=True, time_enc="date",
                 return_att=False):
        super().__init__()
        self.return_att = return_att and temporal == "ltae"
        if temporal == "ltae":
            self.temporal = LTAE(in_channels, use_dates=use_dates, time_enc=time_enc, return_att=self.return_att)
        elif temporal == "mean":
            self.temporal = MeanPool(in_channels)
        else:
            raise ValueError(temporal)
        self.decoder = Decoder(128, num_classes)

    def forward(self, x, days, pad_mask=None):
        if self.return_att:                      # (logits, attention (heads, B, h, w, T))
            z, att = self.temporal(x, days, pad_mask)
            return self.decoder(z), att
        return self.decoder(self.temporal(x, days, pad_mask))
