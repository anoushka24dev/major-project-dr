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
