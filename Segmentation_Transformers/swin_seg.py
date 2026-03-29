"""
Swin Transformer backbone (timm) + multi-scale fusion decoder for semantic segmentation.
Default: ImageNet-1k pretrained Swin backbone (set pretrained=False for from-scratch).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    timm = None
    _TIMM_ERROR = e
else:
    _TIMM_ERROR = None


_BACKBONE_ALIASES = {
    "tiny": "swin_tiny_patch4_window7_224",
    "small": "swin_small_patch4_window7_224",
    "base": "swin_base_patch4_window7_224",
}


def _swin_feat_to_nchw(x: torch.Tensor, channels_out: int) -> torch.Tensor:
    """timm Swin feature tensors may be NCHW or NHWC; Conv2d expects NCHW."""
    if x.ndim != 4:
        return x
    if x.shape[1] == channels_out:
        return x
    if x.shape[-1] == channels_out:
        return x.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(
        f"Unexpected Swin feature shape {tuple(x.shape)} for {channels_out} channels"
    )


class SwinSeg(nn.Module):
    """
    Multi-scale features from Swin (4 stages) → align to H/4 → fuse → predict mask at full res.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: str = "tiny",
        pretrained: bool = True,
        decoder_channels: int = 256,
        img_size: int = 256,
        strict_img_size: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise ImportError(
                "swin_seg requires `timm`. Install with: pip install timm\n"
                f"Original error: {_TIMM_ERROR}"
            )

        name = _BACKBONE_ALIASES.get(variant.lower(), variant)
        isize = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        # Swin in timm defaults to 224×224 with strict H/W check; training uses e.g. 256×256.
        _kw = dict(
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=isize,
            strict_img_size=strict_img_size,
        )
        try:
            self.backbone = timm.create_model(name, **_kw)
        except TypeError:
            _kw.pop("strict_img_size", None)
            try:
                self.backbone = timm.create_model(name, **_kw)
            except TypeError:
                self.backbone = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
                if isize != (224, 224):
                    raise ValueError(
                        "Your timm does not support img_size/strict_img_size on this Swin; "
                        "upgrade timm (`pip install -U timm`) or set IMG_SIZE=224."
                    ) from None
        channels = self.backbone.feature_info.channels()

        self.proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, decoder_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True),
            )
            for c in channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * 4, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.decoder_channels = decoder_channels
        self._backbone_channels = tuple(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(f"SwinSeg expects H,W divisible by 32; got {h}x{w}")

        feats = self.backbone(x)
        h4, w4 = h // 4, w // 4
        aligned = []
        for feat, proj, c_out in zip(feats, self.proj, self._backbone_channels):
            feat = _swin_feat_to_nchw(feat, c_out)
            y = proj(feat)
            y = F.interpolate(y, size=(h4, w4), mode="bilinear", align_corners=False)
            aligned.append(y)

        z = torch.cat(aligned, dim=1)
        z = self.fuse(z)
        z = self.head(z)
        z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
        return z


def swin_seg_tiny(num_classes: int = 1, pretrained: bool = True, **kwargs):
    return SwinSeg(num_classes=num_classes, variant="tiny", pretrained=pretrained, **kwargs)


def swin_seg_small(num_classes: int = 1, pretrained: bool = True, **kwargs):
    return SwinSeg(num_classes=num_classes, variant="small", pretrained=pretrained, **kwargs)
