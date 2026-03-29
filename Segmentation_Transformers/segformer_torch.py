"""
SegFormer in pure PyTorch (no Hugging Face transformers).
MiT encoder + lightweight MLP decoder — matches SegFormer B0–B2 layout from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.0):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.dwconv = nn.Conv2d(mlp_dim, mlp_dim, 3, padding=1, groups=mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(self.fc2(x))
        return x


class MixTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, sr_ratio, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, mlp_ratio, drop)

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class MixTransformerEncoder(nn.Module):
    def __init__(
        self, in_chans, embed_dims, num_heads, depths, sr_ratios, mlp_ratio=4.0, drop_rate=0.0
    ):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        cur = in_chans
        for i in range(len(depths)):
            self.patch_embeds.append(
                OverlapPatchEmbed(
                    cur,
                    embed_dims[i],
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    padding=3 if i == 0 else 1,
                )
            )
            cur = embed_dims[i]
            blocks = nn.ModuleList()
            for _ in range(depths[i]):
                blocks.append(
                    MixTransformerBlock(
                        embed_dims[i],
                        num_heads[i],
                        mlp_ratio,
                        sr_ratios[i],
                        drop_rate,
                        0.0,
                    )
                )
            self.blocks.append(blocks)
        self.norms = nn.ModuleList([nn.LayerNorm(ed) for ed in embed_dims])

    def forward(self, x):
        outs = []
        for i, embed in enumerate(self.patch_embeds):
            x, h, w = embed(x)
            for blk in self.blocks[i]:
                x = blk(x, h, w)
            x = self.norms[i](x)
            x = x.reshape(x.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class SegFormerMLPDecoder(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes, dropout=0.1):
        super().__init__()
        c = len(in_channels)
        self.linear_c = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, embedding_dim, 1),
                    nn.BatchNorm2d(embedding_dim),
                    nn.ReLU(True),
                )
                for ch in in_channels
            ]
        )
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * c, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, feats, target_hw):
        h, w = target_hw
        outs = []
        for i, feat in enumerate(feats):
            x = self.linear_c[i](feat)
            x = F.interpolate(x, size=(h // 4, w // 4), mode="bilinear", align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.linear_fuse(x)
        x = self.classifier(x)
        return x


_VARIANTS = {
    "b0": dict(embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]),
    "b1": dict(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]),
    "b2": dict(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]),
}


class SegFormer(nn.Module):
    """
    SegFormer for semantic segmentation (binary: num_classes=1).
    """

    def __init__(self, num_classes=1, variant="b0", in_chans=3, decoder_dim=256, dropout=0.1):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {list(_VARIANTS.keys())}, got {variant}")
        cfg = _VARIANTS[variant]
        self.encoder = MixTransformerEncoder(
            in_chans=in_chans,
            embed_dims=cfg["embed_dims"],
            num_heads=cfg["num_heads"],
            depths=cfg["depths"],
            sr_ratios=cfg["sr_ratios"],
            mlp_ratio=4.0,
            drop_rate=dropout,
        )
        self.decoder = SegFormerMLPDecoder(
            cfg["embed_dims"], decoder_dim, num_classes, dropout=dropout
        )

    def forward(self, x):
        target_hw = (x.shape[2], x.shape[3])
        feats = self.encoder(x)
        logits = self.decoder(feats, target_hw)
        return logits


def segformer_b0(num_classes=1, **kwargs):
    return SegFormer(num_classes=num_classes, variant="b0", **kwargs)


def segformer_b1(num_classes=1, **kwargs):
    return SegFormer(num_classes=num_classes, variant="b1", **kwargs)


def segformer_b2(num_classes=1, **kwargs):
    return SegFormer(num_classes=num_classes, variant="b2", **kwargs)
