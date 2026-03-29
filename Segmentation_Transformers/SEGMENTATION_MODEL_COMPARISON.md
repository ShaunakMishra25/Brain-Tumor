# BRISC segmentation — model comparison report

**Dataset:** `BRISC/archive/brisc2025/segmentation_task` (binary masks, **256×256** in these experiments).  
**Metrics:** Same style as each notebook’s test loop (mean Dice, IoU, precision, recall, accuracy). Values are from **saved notebook outputs** where available — single runs, not cross‑validated. Re-run notebooks after config changes to refresh numbers.

## Summary table (ranked by Mean Dice)

| Rank | Model | Notebook / code | Dice | IoU | Precision | Recall | Accuracy |
|------|--------|-----------------|------|-----|-----------|--------|----------|
| 1 | **SwinSeg** (Swin‑T, ImageNet backbone via timm + fusion decoder) | `Segmentation_Transformers/SwinSeg.ipynb` · `swin_seg.py` | **0.8765** | **0.8088** | 0.8828 | 0.8936 | 0.9965 |
| 2 | **DeepLabV3+** (ResNet‑50 backbone) | `DeepLabV3plus/DeepLabV3ii.ipynb` | 0.8568 | 0.7815 | 0.8772 | 0.8658 | 0.9961 |
| 3 | DeepLabV3+ (EfficientNet‑B0) | `DeepLabV3plus/DeepLabV3EfficientNetB0.ipynb` | 0.8566 | 0.7798 | 0.8482 | 0.8928 | 0.9960 |
| 4 | DeepLabV3+ (MobileNet) | `DeepLabV3plus/DeepLabV3MobileNet.ipynb` | 0.8432 | 0.7659 | 0.8486 | 0.8721 | 0.9959 |
| 5 | **SegFormer** (MiT‑B0, custom PyTorch — train encoder from scratch) | `Segmentation_Transformers/SegFormer.ipynb` · `segformer_torch.py` | 0.8273 | 0.7439 | 0.8370 | 0.8533 | 0.9951 |
| 6 | UNet++ | `UNetpp/UNetpp.ipynb` | 0.8178 | 0.7460 | — | — | — |
| 7 | UNet (baseline) | `UNet/UNet.ipynb` | 0.8080 | 0.7341 | — | — | — |
| 8 | **SETR** (ViT encoder + PUP decoder; see notebook for ViT init) | `Segmentation_Transformers/SETR.ipynb` | 0.7929 | 0.7030 | 0.8407 | 0.7853 | 0.9951 |
| 9 | UNet + augmentation | `UNet/UNetWAugmentation.ipynb` | 0.7712 | 0.6890 | — | — | — |
| 10 | Attention U‑Net | `Attention Unet/AUnet.ipynb` | 0.7652 | 0.6803 | — | — | — |

*“—”* = precision/recall/accuracy not transcribed from that notebook’s saved output (metrics cell exists — open file to confirm).

## Segmentation transformers folder (`Segmentation_Transformers/`)

| File | Role |
|------|------|
| `SegFormer.ipynb` / `segformer_torch.py` | MiT + MLP decoder, **no** external timm/HF; encoder scratch unless you change it. |
| `SETR.ipynb` | ViT‑style encoder + Naive / PUP / MLA decoders. |
| `SwinSeg.ipynb` / `swin_seg.py` | **Swin** backbone (**timm**, ImageNet pretrained by default) + 4‑stage fusion decoder. |
| `SETR.ipynb` / `SegFormer.ipynb` / `SwinSeg.ipynb` | Each writes predictions under `archive/.../segmentation_task/predictions/<ModelName>/`. |

## Training / methodology (high level)

| Aspect | DeepLabV3+ (`DeepLabV3ii`) | SegFormer | SETR | SwinSeg |
|--------|----------------------------|-----------|------|---------|
| Encoder init | ImageNet ResNet‑50 | **Scratch** MiT (`segformer_torch.py`) | ViT‑style (notebook) | **ImageNet** Swin (**timm**) |
| Typical epochs | 25 | 40 + warmup/cosine | 25 (check notebook) | 40 + warmup/cosine |
| Loss | BCE | BCE + soft Dice | BCE | BCE + soft Dice |
| Aug (train) | Resize + ToTensor | Paired flips + ColorJitter + ImageNet norm | Basic (check notebook) | Same style as SegFormer (flips + jitter + norm) |

**Why SwinSeg can beat scratch SegFormer:** pretrained hierarchical backbone + strong aug/schedule, comparable to the advantage DeepLab gets from pretrained ResNet‑50.

## Prediction & artifact paths

| Model | Base folder under `.../segmentation_task/predictions/` |
|--------|----------------------------------------------------------|
| DeepLabV3+ (ii) | `DeepLabV3/` |
| SegFormer | `SegFormer/` |
| SETR | `SETR/` |
| SwinSeg | `SwinSeg/` |

Checkpoints often live in `predictions/models/` with names like `segformer_*_best.pth`, `swinseg_*_best.pth`.

## Historical note (SegFormer)

An earlier SegFormer run (~**0.8044** Dice / **0.7193** IoU) improved to **0.8273** / **0.7439** after augmentation + AdamW + LR schedule (still scratch MiT).

---

*Update this file after major runs, or replace table cells with your latest notebook prints.*
