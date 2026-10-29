"""The 10-band student for Experiment 2 (distillation).

Student = DINOv3 ViT-S/16 (web weights by default) whose first layer is replaced so it
reads all 10 Sentinel-2 bands. The RGB part of the new first layer is copied from the
original (B4 <- R, B3 <- G, B2 <- B); the 7 other bands start at zero, so at the start
the student behaves like the RGB model.

Input: 10 bands, standardised per band, upsampled 128 -> 256 px  ->  16 x 16 tokens of 384.

Training heads (only used during distillation):
  * projector W: 384 -> teacher dim (1024), for the copy loss (cosine distance)
  * rebuild decoder r: token (384) -> the 7 non-RGB bands of its 8 x 8 native pixels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino import num_prefix_tokens
from .pastis import NON_RGB_IDX, RGB_IDX


def make_student_backbone(init, in_channels=10, random_config=None):
    """init: a hub id / local folder with DINOv3 weights, or "random" (then random_config is used)."""
    from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel

    if init == "random":
        cfg = DINOv3ViTConfig(**(random_config or dict(
            hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
            intermediate_size=1536, patch_size=16, num_register_tokens=4)))
        model = DINOv3ViTModel(cfg)
    else:
        model = AutoModel.from_pretrained(init)

    old = model.embeddings.patch_embeddings
    assert isinstance(old, nn.Conv2d) and old.in_channels == 3, "unexpected patch embedding"
    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        for rgb_ch, band in enumerate(RGB_IDX):        # R <- B4, G <- B3, B <- B2
            new.weight[:, band] = old.weight[:, rgb_ch]
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.embeddings.patch_embeddings = new
    model.config.num_channels = in_channels
    return model


class Student(nn.Module):
    def __init__(self, backbone, teacher_dim=1024, input_size=256, native_size=128):
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        d = backbone.config.hidden_size
        self.dim = d
        self.patch = backbone.config.patch_size
        self.native_px = self.patch * native_size // input_size     # native pixels per token side (8)
        n_out = len(NON_RGB_IDX) * self.native_px ** 2
        self.projector = nn.Linear(d, teacher_dim)
        self.rebuild = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_out))

    def tokens(self, x):
        """x: (B, 10, 128, 128) standardised bands -> (B, D, h, w) tokens."""
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear",
                              align_corners=False)
        out = self.backbone(pixel_values=x).last_hidden_state
        t = out[:, num_prefix_tokens(self.backbone):, :]
        B, N, D = t.shape
        h = w = self.input_size // self.patch
        return t.transpose(1, 2).reshape(B, D, h, w)

    def forward(self, x, detach_copy=False, detach_rebuild=False):
        """detach_*: train that head on frozen (detached) tokens, so it only *measures*
        what the tokens contain and does not change the student. Used for student 1, whose
        rebuild head then shows how much infrared information its tokens still hold."""
        tok = self.tokens(x)                                   # B, D, h, w
        B, D, h, w = tok.shape
        seq = tok.flatten(2).transpose(1, 2)                   # B, N, D
        proj = self.projector(seq.detach() if detach_copy else seq)        # B, N, teacher_dim
        rec = self.rebuild(seq.detach() if detach_rebuild else seq)        # B, N, 7*p*p
        p = self.native_px
        rec = rec.view(B, h, w, len(NON_RGB_IDX), p, p).permute(0, 3, 1, 4, 2, 5)
        rec = rec.reshape(B, len(NON_RGB_IDX), h * p, w * p)  # B, 7, 128, 128
        return tok, proj, rec


def student_from_checkpoint(ck):
    """Rebuild the student from a best.pt checkpoint without downloading anything."""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    cfg = DINOv3ViTConfig(**ck["backbone_config"])
    model = Student(DINOv3ViTModel(cfg), teacher_dim=ck["teacher_dim"])
    model.load_state_dict(ck["model"])
    return model


def copy_loss(proj, teacher_tokens):
    """Mean over patches of 1 - cosine(student projected, teacher). teacher_tokens: (B, N, Dt)."""
    return (1 - F.cosine_similarity(proj.float(), teacher_tokens.float(), dim=-1)).mean()


def rebuild_loss(rec, target):
    """MSE between rebuilt and true (standardised) non-RGB bands."""
    return F.mse_loss(rec.float(), target.float())
