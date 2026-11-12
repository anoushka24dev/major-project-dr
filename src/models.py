"""Model definitions: five-class ResNet-50 and the four-channel lesion U-Net.

There is exactly **one** ResNet-50 builder in the unified pipeline (this module).
The legacy Mendeley-only baseline in ``colab/colab_helpers.py`` keeps its own
frozen-backbone builder and is untouched; both load the same
``ResNet50_Weights.IMAGENET1K_V1`` weights, so head shapes and initialisation are
comparable (asserted by ``tests/test_unified_pipeline.py``).

The U-Net's output channel order is fixed by the project: **MA, HE, EX, SE**.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# ResNet-50
# ---------------------------------------------------------------------------

def build_resnet50(
    num_classes: int = 5,
    *,
    pretrained: bool = True,
    allow_weight_download: bool = True,
    freeze_backbone: bool = False,
    logger=None,
) -> nn.Module:
    """Build a ResNet-50 with a fresh ``num_classes`` head.

    Parameters
    ----------
    pretrained:
        Load ImageNet weights. When they are not cached locally and
        ``allow_weight_download`` is False, this raises instead of starting a
        silent network transfer.
    freeze_backbone:
        Freeze every parameter except the new head (used for the head-only warm-up
        epochs configured by ``classification.freeze_backbone_epochs``).
    """
    import torchvision.models as models

    weights = None
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        if not allow_weight_download:
            try:
                from .legacy_bridge import load_colab_helpers
                from .config import project_root

                helpers = load_colab_helpers(project_root())
                if not helpers.pretrained_weights_available():
                    raise FileNotFoundError(
                        "ImageNet ResNet-50 weights are not cached locally at "
                        f"{helpers.weights_cache_path()}. Allow the download or fetch them once."
                    )
            except FileNotFoundError:
                raise
            except Exception:
                pass  # helper unavailable: torchvision will raise its own clear error

    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, int(num_classes))

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    if logger:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "ResNet-50 built: %d classes, pretrained=%s, frozen=%s, trainable=%s/%s params",
            num_classes, bool(pretrained), freeze_backbone, f"{trainable:,}", f"{total:,}",
        )
    return model


def unfreeze_all(model: nn.Module) -> None:
    """Make every parameter trainable (end of the head-only warm-up)."""
    for param in model.parameters():
        param.requires_grad = True


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the final convolutional block of a ResNet-50 (``layer4[-1]``)."""
    layer4 = getattr(model, "layer4", None)
    if layer4 is None:
        raise ValueError("Model has no layer4; expected a torchvision ResNet.")
    return layer4[-1]


# ---------------------------------------------------------------------------
# Four-channel U-Net
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


SUPPORTED_CLASSIFIER_ARCHITECTURES = (
    "resnet50",
    "resnet50_v2",
    "resnet101",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
)


class GeM2d(nn.Module):
    """Generalized Mean Pooling for fine lesion localization in retinal feature maps."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.p, min=1.0, max=8.0)
        shift = torch.clamp(-x.amin(dim=(-2, -1), keepdim=True), min=0.0)
        x_pos = (x + shift).clamp(min=self.eps)
        return F.adaptive_avg_pool2d(x_pos.pow(p), (1, 1)).pow(1.0 / p) - shift


def build_classifier(
    architecture: str = "resnet50",
    num_classes: int = 5,
    *,
    pretrained: bool = True,
    allow_weight_download: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
    pooling: str = "avg",
    logger=None,
) -> nn.Module:
    """Build a five-class retinal classifier for the named torchvision backbone."""
    arch = str(architecture or "resnet50").strip().lower()
    pool_mode = str(pooling or "avg").strip().lower()
    if pool_mode not in {"avg", "gem"}:
        raise ValueError(f"Unsupported pooling mode {pooling!r}; choose 'avg' or 'gem'.")
    if arch == "resnet50" and float(dropout or 0.0) <= 0.0 and pool_mode == "avg":
        return build_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            allow_weight_download=allow_weight_download,
            freeze_backbone=freeze_backbone,
            logger=logger,
        )

    from torchvision import models

    weight_map = {
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
        "resnet50_v2": models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet101": models.ResNet101_Weights.IMAGENET1K_V2,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_small": models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "convnext_base": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "efficientnet_v2_m": models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        "swin_v2_t": models.Swin_V2_T_Weights.IMAGENET1K_V1,
        "swin_v2_s": models.Swin_V2_S_Weights.IMAGENET1K_V1,
        "swin_v2_b": models.Swin_V2_B_Weights.IMAGENET1K_V1,
    }
    builder_map = {
        "resnet50": models.resnet50,
        "resnet50_v2": models.resnet50,
        "resnet101": models.resnet101,
        "convnext_tiny": models.convnext_tiny,
        "convnext_small": models.convnext_small,
        "convnext_base": models.convnext_base,
        "efficientnet_v2_s": models.efficientnet_v2_s,
        "efficientnet_v2_m": models.efficientnet_v2_m,
        "swin_v2_t": models.swin_v2_t,
        "swin_v2_s": models.swin_v2_s,
        "swin_v2_b": models.swin_v2_b,
    }
    if arch not in builder_map:
        raise ValueError(
            f"Unsupported classifier architecture {architecture!r}. "
            f"Choose one of {SUPPORTED_CLASSIFIER_ARCHITECTURES}."
        )

    weights = None
    if pretrained:
        if not allow_weight_download:
            raise RuntimeError(
                f"pretrained weights are requested for {arch} but allow_weight_download is False. "
                "Set allow_weight_download: true in config.yaml or pre-cache torchvision weights."
            )
        weights = weight_map[arch]

    try:
        model = builder_map[arch](weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                f"Failed to obtain ImageNet weights for {arch}: {exc}."
            ) from exc
        raise

    drop = float(dropout or 0.0)
    if arch in {"resnet50", "resnet50_v2", "resnet101"}:
        in_features = model.fc.in_features
        head: nn.Module = nn.Linear(in_features, num_classes)
        model.fc = nn.Sequential(nn.Dropout(p=drop), head) if drop > 0 else head
        head_params = set(model.fc.parameters())
    elif arch in {"convnext_tiny", "convnext_small", "convnext_base"}:
        in_features = model.classifier[2].in_features
        head = nn.Linear(in_features, num_classes)
        model.classifier[2] = nn.Sequential(nn.Dropout(p=drop), head) if drop > 0 else head
        head_params = set(model.classifier[2].parameters())
    elif arch in {"efficientnet_v2_s", "efficientnet_v2_m"}:
        in_features = model.classifier[1].in_features
        if drop > 0:
            model.classifier[0] = nn.Dropout(p=drop, inplace=True)
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head_params = set(model.classifier.parameters())
    else:
        # swin_v2_t / swin_v2_s / swin_v2_b
        in_features = model.head.in_features
        head = nn.Linear(in_features, num_classes)
        model.head = nn.Sequential(nn.Dropout(p=drop), head) if drop > 0 else head
        head_params = set(model.head.parameters())

    if pool_mode == "gem" and hasattr(model, "avgpool"):
        model.avgpool = GeM2d(p=3.0)
        head_params = head_params | set(model.avgpool.parameters())

    if freeze_backbone:
        for param in model.parameters():
            if param not in head_params:
                param.requires_grad = False

    if logger:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "%s built: pretrained=%s, pooling=%s, freeze_backbone=%s, dropout=%.2f, total=%s, trainable=%s params",
            arch, pretrained, pool_mode, freeze_backbone, drop, f"{total:,}", f"{trainable:,}",
        )
    return model


@torch.no_grad()
def predict_probs_with_tta(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    tta: object = False,
    device=None,
    amp_enabled: bool = False,
) -> torch.Tensor:
    """Run forward inference with optional fundus-preserving test-time augmentation."""
    if tta is True:
        mode = "hflip"
    elif not tta:
        mode = "none"
    else:
        mode = str(tta).strip().lower()
    if mode in {"false", "off", "0", ""}:
        mode = "none"

    dev = device if device is not None else inputs.device
    use_amp = bool(amp_enabled and dev.type == "cuda")
    x = inputs.to(dev, non_blocking=True)

    views = [x]
    if mode in {"hflip", "multi"}:
        views.append(torch.flip(x, dims=[-1]))
    if mode == "multi":
        _, _, h, w = x.shape
        margin_h = max(1, int(round(h * 0.03)))
        margin_w = max(1, int(round(w * 0.03)))
        if h - 2 * margin_h >= 16 and w - 2 * margin_w >= 16:
            cropped = x[:, :, margin_h : h - margin_h, margin_w : w - margin_w]
            zoomed = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
            views.append(zoomed)
            views.append(torch.flip(zoomed, dims=[-1]))

    prob_sum = None
    with torch.amp.autocast(device_type=dev.type, enabled=use_amp):
        for view in views:
            probs = torch.softmax(model(view), dim=1).float()
            prob_sum = probs if prob_sum is None else (prob_sum + probs)
    return prob_sum / float(len(views))


class UNet4(nn.Module):
    """U-Net producing four independent lesion logit maps (MA, HE, EX, SE).

    The four channels are *not* softmaxed against each other: lesions co-occur, so
    each channel is an independent binary problem with its own sigmoid.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 4, base_filters: int = 32) -> None:
        super().__init__()
        b = int(base_filters)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        self.head = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        bottom = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.dec4(torch.cat([self.up4(bottom), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities ``(B, 4, H, W)``."""
        return torch.sigmoid(self.forward(x))


def build_unet4(
    *, in_channels: int = 3, out_channels: int = 4, base_filters: int = 32, logger=None
) -> UNet4:
    """Build the four-channel lesion U-Net."""
    model = UNet4(in_channels=in_channels, out_channels=out_channels, base_filters=base_filters)
    if logger:
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "U-Net built: in=%d out=%d base=%d total=%s params",
            in_channels, out_channels, base_filters, f"{total:,}",
        )
    return model


# ---------------------------------------------------------------------------
# Segmentation loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Weighted sum of binary cross-entropy and soft Dice loss.

    BCE stabilises early gradients on the huge background class; Dice directly
    optimises the overlap that the evaluation reports, and is insensitive to the
    extreme foreground sparsity of microaneurysms.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.register_buffer("bce", torch.tensor(0.0), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dims)
        denominator = probs.sum(dims) + targets.sum(dims)
        dice_per_channel = (2 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice_per_channel.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def model_summary(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
    """Run one forward pass on a zero tensor and report the output shape."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(*input_shape))
    if was_training:
        model.train()
    return {
        "class": type(model).__name__,
        "output_shape": list(output.shape),
        "parameters": count_parameters(model),
    }
