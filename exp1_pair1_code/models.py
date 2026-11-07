"""Temporal models on frozen DINOv3 features + a shared decoder.

L-TAE is the OFFICIAL implementation from the U-TAE / PASTIS repo
(github.com/VSainteuf/utae-paps), copied unchanged into src/backbones/.
Run from inside exp1/ so `src.backbones` can be imported.

All variants share the same first layers (GroupNorm + 1x1 projection 1024 -> 256,
as inside LTAE2d) and the same decoder. Only the temporal step differs:
  agg='single'  one date
  agg='mean'    average over the dates it is given (padding excluded)
  agg='ltae'    LTAE2d, with the time input switched by pe:
                none | position | date | both | shuffled
"""
import torch
import torch.nn as nn

from src.backbones.ltae import LTAE2d
from src.backbones.positional_encoding import PositionalEncoder

PACK = 10000.0   # 'both': position and day packed into one number, pos*PACK + day


class BothEncoder(nn.Module):
    """For pe='both': in every head, half the dims encode position, half encode the date.

    LTAE2d.forward passes a single [N, T] tensor to its positional encoder, so position and
    day arrive packed as pos*PACK + day (exact in float32: max ~6e6 < 2^24) and are unpacked here.
    """
    def __init__(self, d_head, T=1000, repeat=16):
        super().__init__()
        self.pos = PositionalEncoder(d_head // 2, T=T)
        self.day = PositionalEncoder(d_head - d_head // 2, T=T)
        self.repeat = repeat

    def forward(self, packed):
        pos = torch.div(packed, PACK, rounding_mode="floor")
        day = packed - pos * PACK
        one_head = torch.cat([self.pos(pos), self.day(day)], dim=-1)
        return torch.cat([one_head] * self.repeat, dim=-1)


class Decoder(nn.Module):
    """16x16 -> 128x128 in three x2 steps."""
    def __init__(self, d_in=256, num_classes=20, widths=(256, 128, 64)):
        super().__init__()
        layers, c = [], d_in
        for w in widths:
            layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, z):
        return self.head(self.body(z))


class TemporalSegmenter(nn.Module):
    def __init__(self, d_feat=1024, d_model=256, agg="ltae", pe="date", num_classes=20,
                 n_head=16, d_k=4):
        super().__init__()
        self.agg, self.pe = agg, pe
        if agg == "ltae":
            self.ltae = LTAE2d(in_channels=d_feat, n_head=n_head, d_k=d_k,
                               mlp=[d_model, d_model], d_model=d_model,
                               return_att=True, positional_encoding=(pe != "none"))
            if pe == "both":
                self.ltae.positional_encoder = BothEncoder(d_model // n_head, repeat=n_head)
        else:
            # same as the first two layers inside LTAE2d (in_norm + inconv)
            self.in_norm = nn.GroupNorm(n_head, d_feat)
            self.in_proj = nn.Conv2d(d_feat, d_model, 1)
        self.decoder = Decoder(d_model, num_classes)

    def time_input(self, pos, days):
        if self.pe == "none":
            return None
        if self.pe == "position":
            return pos
        if self.pe in ("date", "shuffled"):
            return days
        if self.pe == "both":
            return pos * PACK + days
        raise ValueError(self.pe)

    def forward(self, x, pos, days, pad):
        """x: [B,T,D,h,w], pos/days: [B,T], pad: [B,T] True = padding.
        Returns logits [B,C,128,128] and attention [heads,B,T,h,w] (None if not L-TAE)."""
        x = x.float()
        if self.agg == "ltae":
            z, attn = self.ltae(x, batch_positions=self.time_input(pos.float(), days.float()),
                                pad_mask=pad)
            return self.decoder(z), attn

        B, T, D, h, w = x.shape
        z = self.in_proj(self.in_norm(x.reshape(B * T, D, h, w))).view(B, T, -1, h, w)
        if self.agg == "single":
            z = z[:, 0]
        else:  # mean over real dates only
            m = (~pad).float()[:, :, None, None, None]
            z = (z * m).sum(1) / m.sum(1).clamp(min=1)
        return self.decoder(z), None
