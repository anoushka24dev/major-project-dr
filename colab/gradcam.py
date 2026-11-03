"""Grad-CAM explainability for the frozen ResNet-50 primary baseline.

What this module does
---------------------
This is a *read-only*, inference-only explainability stage. It:

- loads the existing validation-selected checkpoint in ``<run>/best_checkpoint``;
- rebuilds the identical architecture with ``pretrained=False`` (so no ImageNet
  download is attempted) and restores the trained weights with the same loader the
  final-evaluation path already uses;
- computes class-discriminative Grad-CAM maps from the last convolutional stage of
  ResNet-50 (``layer4[-1]``, i.e. ``model.layer4[2]``, the final bottleneck block);
- selects confidently-correct and confidently-wrong examples per band;
- writes PNG overlays plus one montage per band;
- writes ``gradcam_report.json`` (machine-readable) and ``GRADCAM_REPORT.md``
  (attention-pattern summary written for a human reader).

What this module does NOT do
----------------------------
- it does not train, fine-tune, or resume anything: there is no optimizer, no
  ``loss.backward()`` into parameters, and no weight update;
- it does not write, move, or delete any checkpoint, epoch directory, run
  manifest, training state, or log inside the run;
- it does not touch the dataset, the split CSV, or the split report;
- it does not change the label mapping or the preprocessing used at training time;
- it makes no clinical claim. Grad-CAM is a low-resolution (7x7) saliency
  approximation, not lesion segmentation, and the five-band mapping is provisional.

Why ``layer4[-1]``
------------------
Grad-CAM needs the last layer that still carries *spatial* information. In this
ResNet-50 the sequence after ``layer4`` is ``avgpool -> fc``: ``avgpool`` collapses
7x7x2048 to 1x1x2048, so everything downstream of it has no spatial extent to
weight. ``layer4[-1]`` is therefore the last convolutional output whose activations
can be turned into a 7x7 map that is then upsampled to the input resolution. Its
receptive field covers the full fundus, and its channels encode the high-level
DR-relevant patterns (microaneurysms/haemorrhages/exudates and their arrangement)
rather than edge/texture primitives from earlier stages.

Usage
-----
    python -m colab.gradcam --output-dir cpu_runs/full_restart2
    python -m colab.gradcam --output-dir cpu_runs/full_restart2 --partition test --confirm
    python -m colab.gradcam --output-dir cpu_runs/full_restart2 --per-class 3 --max-scan 800

The ``test`` partition is guarded by ``--confirm`` for the same reason the final
evaluation is: the held-out rows are not something to wander into by accident.
The default partition is ``validation``, which is already visible to the model
selection step and is therefore the cheaper, lower-consequence place to look.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "colab")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import colab.setup_helpers as sh  # noqa: E402
import colab.train as train_mod  # noqa: E402
from colab_helpers import (  # noqa: E402
    BAND_ORDER,
    build_frozen_resnet50,
    default_normalize,
    filter_by_partition,
    load_rgb,
    load_split_csv,
    preprocessing_inference,
)

IMAGENET_MEAN, IMAGENET_STD = default_normalize()

EXPLAIN_DIR_NAME = "explainability"
GRADCAM_DIR_NAME = "gradcam"
REPORT_MD_NAME = "GRADCAM_REPORT.md"
REPORT_JSON_NAME = "gradcam_report.json"

#: ResNet-50 stage recommended for Grad-CAM (last stage before global pooling).
CAM_STAGE_ATTR = "layer4"
CAM_LAYER_DISPLAY = "layer4[-1] (model.layer4[2], final bottleneck block)"

PARTITION_CHOICES = ("validation", "test")


# ---------------------------------------------------------------------------
# Small tensor helpers (mirrors colab.train, which owns the dataset contract)
# ---------------------------------------------------------------------------

def _to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to float32 CHW in [0, 1]."""
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # defensive: single-channel image
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def _normalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Apply per-channel ImageNet normalization (out-of-place)."""
    m = torch.as_tensor(list(mean), dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    s = torch.as_tensor(list(std), dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - m) / s


def load_model_input(path: Path, image_size: int = 224) -> Tuple[torch.Tensor, Image.Image]:
    """Return ``(normalized CHW tensor, preprocessed RGB image)`` for one file.

    Uses the same functions as ``colab.train.PrimarySplitDataset``: deterministic
    aspect-ratio-preserving resize-and-pad, then ImageNet normalization. No center
    crop and no random crop, exactly as at training and evaluation time.
    """
    base = preprocessing_inference(image_size)(load_rgb(Path(path)))
    tensor = _normalize(_to_tensor(base))
    return tensor, base


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def _module_name(model: torch.nn.Module, target: torch.nn.Module) -> str:
    """Return the dotted module name for ``target``, or ``'<unknown>'``."""
    for name, module in model.named_modules():
        if module is target:
            return name
    return "<unknown>"


def last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last ``Conv2d`` in module-registration order."""
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("Model contains no Conv2d layer, so Grad-CAM is undefined.")
    return last


def resolve_cam_layer(
    model: torch.nn.Module,
    layer: Optional[torch.nn.Module] = None,
) -> Tuple[str, torch.nn.Module]:
    """Return ``(module_name, module)`` for the Grad-CAM target.

    For a ResNet-50 this is ``layer4[-1]``. For anything else, the last ``Conv2d``
    is used as a documented fallback rather than silently pretending the map came
    from a high-level stage.
    """
    if layer is not None:
        return _module_name(model, layer), layer

    stage = getattr(model, CAM_STAGE_ATTR, None)
    if stage is not None and len(stage) > 0:
        target = stage[-1]
    else:
        target = last_conv_layer(model)
    return _module_name(model, target), target


def layer_facts(model: torch.nn.Module, name: str, layer: torch.nn.Module, image_size: int) -> Dict[str, Any]:
    """Describe the chosen layer so the report can be audited later."""
    convs = [m for m in layer.modules() if isinstance(m, torch.nn.Conv2d)]
    out_channels = None
    for attr in ("out_channels",):
        out_channels = getattr(layer, attr, None)
        if out_channels is not None:
            break
    if out_channels is None and convs:
        out_channels = convs[-1].out_channels

    # ResNet-50 downsamples by 32 overall, so a 224 px input gives a 7x7 map.
    stride_total = 32
    spatial = max(1, image_size // stride_total)
    return {
        "module_name": name or CAM_LAYER_DISPLAY,
        "module_type": type(layer).__name__,
        "out_channels": out_channels,
        "conv_layers_inside": len(convs),
        "spatial_map": f"{spatial}x{spatial} for a {image_size}x{image_size} input",
        "downsample_factor": stride_total,
        "justification": (
            "Grad-CAM needs the last layer that still carries spatial extent. After "
            f"{CAM_STAGE_ATTR} the network applies avgpool, which collapses all spatial "
            "information to 1x1 before fc, so any hook downstream of avgpool has no "
            "positional axis to weight. The final bottleneck block of layer4 is the "
            "deepest spatially-resolved stage: its receptive field covers the whole "
            "fundus and its channels encode high-level DR patterns rather than the "
            "edge/texture primitives of the earlier stages."
        ),
    }


# ---------------------------------------------------------------------------
# Grad-CAM engine
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted class activation mapping for one target layer.

    Per image: hook the target layer's forward output, run a normal forward pass
    with ``requires_grad`` enabled on the input, take ``d logit[class] / d A``, then
    ``cam = ReLU(sum_k mean_ij(dY/dA)_k * A_k)`` and upsample to input resolution.

    No parameters are updated. ``torch.autograd.grad`` needs the activations to
    require grad, which they do as soon as the *input* does - all backbone weights
    in this baseline are frozen, so that is the only way gradients reach the layer.

    Note on the backward pass: global average pooling over the gradient is the
    Grad-CAM definition, so the map is 7x7 by construction and is faithful to the
    method, not to lesion geometry. Do not read individual pixels as lesions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, device=None):
        self.model = model
        self.target_layer = target_layer
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        # Grad-CAM must run in eval mode: BatchNorm in train mode would both change
        # the activations and update running statistics, which is a (small) mutation
        # of the loaded model and would make the maps irreproducible.
        self.model.eval()

    def __call__(self, inputs: torch.Tensor, class_indices: Sequence[int]) -> np.ndarray:
        """Return a ``(B, H, W)`` float32 array of maps in [0, 1]."""
        if inputs.ndim != 4:
            raise ValueError(f"Expected a 4-D input tensor, got shape {tuple(inputs.shape)}.")
        batch = int(inputs.shape[0])
        indices = [int(i) for i in class_indices]
        if len(indices) != batch:
            raise ValueError(
                f"Got {len(indices)} class indices for a batch of {batch} images."
            )

        x = inputs.to(self.device).detach().float().requires_grad_(True)
        captured: Dict[str, torch.Tensor] = {}

        def _hook(_module, _inp, output):
            captured["value"] = output

        handle = self.target_layer.register_forward_hook(_hook)
        try:
            with torch.enable_grad():
                logits = self.model(x)
        finally:
            handle.remove()

        activation = captured.get("value")
        if activation is None:
            raise RuntimeError(
                "The Grad-CAM hook never fired: the target layer is not on this model's "
                "forward path."
            )
        if not activation.requires_grad:
            raise RuntimeError(
                "Target-layer activations do not require grad, so no CAM gradient exists. "
                "This happens when the forward pass ran under torch.no_grad()."
            )

        idx = torch.as_tensor(indices, dtype=torch.long, device=logits.device).view(-1, 1)
        score = logits.gather(1, idx).sum()
        grads = torch.autograd.grad(score, activation)[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(int(inputs.shape[-2]), int(inputs.shape[-1])),
            mode="bilinear", align_corners=False,
        )
        cam = cam.detach().cpu().numpy()[:, 0].astype(np.float32)

        out = np.zeros_like(cam, dtype=np.float32)
        for i in range(cam.shape[0]):
            peak = float(cam[i].max())
            if peak > 0:
                out[i] = cam[i] / peak
        return out


# ---------------------------------------------------------------------------
# Overlay rendering (PIL + numpy only, no plotting dependency)
# ---------------------------------------------------------------------------

def _jet_lut(steps: int = 256) -> np.ndarray:
    """Return a ``(steps, 3)`` uint8 jet-like colormap."""
    x = np.linspace(0.0, 1.0, steps)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    lut = np.stack([r, g, b], axis=-1)
    lut[0] = 0.0  # zero activation stays neutral instead of dark blue
    return (lut * 255.0).astype(np.uint8)


_JET_LUT = _jet_lut()


def render_overlay(base_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a normalized CAM onto an RGB uint8 image, alpha-scaled by activation."""
    if base_rgb.shape[:2] != cam.shape:
        raise ValueError(
            f"CAM shape {cam.shape} does not match image shape {base_rgb.shape[:2]}."
        )
    idx = np.clip((np.clip(cam, 0.0, 1.0) * 255.0).astype(np.int32), 0, 255)
    heat = _JET_LUT[idx].astype(np.float32)
    weight = (float(alpha) * np.clip(cam, 0.0, 1.0))[..., None]
    blended = base_rgb.astype(np.float32) * (1.0 - weight) + heat * weight
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def fundus_mask(base_rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of the retina area, i.e. everything that is not padding."""
    gray = base_rgb.astype(np.float32).mean(axis=-1)
    mask = gray > 12.0
    if mask.mean() < 0.2:  # pathological image: treat the whole frame as content
        return np.ones(mask.shape, dtype=bool)
    return mask


def attention_stats(cam: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Summarize where a single CAM puts its mass.

    ``fundus_mass`` is the share of activation inside the retina rather than the
    black pad; ``top20_mass`` is how concentrated the map is; the centroid is in
    [-1, 1] coordinates and ``centroid_radius`` is its distance from image centre.
    """
    total = float(cam.sum())
    h, w = cam.shape
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    if total <= 0:
        return {
            "fundus_mass": 1.0,
            "top20_mass": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "centroid_radius": 0.0,
            "peak_x": 0.0,
            "peak_y": 0.0,
        }

    mass = cam / total
    fundus_mass = float(mass[mask].sum())
    flat = np.sort(cam.ravel())[::-1]
    keep = max(1, int(round(flat.size * 0.2)))
    top20_mass = float(flat[:keep].sum() / total)

    cx = float((mass * xs).sum()) / max(w - 1, 1) * 2.0 - 1.0
    cy = float((mass * ys).sum()) / max(h - 1, 1) * 2.0 - 1.0
    peak_index = int(np.argmax(cam))
    peak_y, peak_x = divmod(peak_index, w)

    return {
        "fundus_mass": fundus_mass,
        "top20_mass": top20_mass,
        "centroid_x": cx,
        "centroid_y": cy,
        "centroid_radius": float(min(1.0, (cx * cx + cy * cy) ** 0.5 / (2.0 ** 0.5))),
        "peak_x": peak_x / max(w - 1, 1) * 2.0 - 1.0,
        "peak_y": peak_y / max(h - 1, 1) * 2.0 - 1.0,
    }


def _slug(text: str) -> str:
    """Filesystem-safe slug for band names and image stems."""
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch.lower())
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "item"


# ---------------------------------------------------------------------------
# Scanning a partition (inference only)
# ---------------------------------------------------------------------------

def _row_dicts(batch: Any) -> List[Dict[str, Any]]:
    """Normalize a collated row batch into one dict per sample.

    ``torch.utils.data.default_collate`` turns a batch of ``Dict[str, str]`` into a
    ``Dict[str, List[str]]``, so indexing it per sample needs transposing back.
    """
    if isinstance(batch, dict):
        keys = list(batch.keys())
        values = [batch[k] for k in keys]
        if not values:
            return []
        return [dict(zip(keys, sample)) for sample in zip(*values)]
    return list(batch)


def scan_partition(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    *,
    batch_size: int = 32,
    progress_every: int = 10,
) -> List[Dict[str, Any]]:
    """Run one no-grad pass and record the per-image prediction outcome."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    records: List[Dict[str, Any]] = []
    model.eval()
    seen = 0
    started = time.time()
    with torch.no_grad():
        for step, (inputs, labels, batch_rows) in enumerate(loader, start=1):
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1).detach().cpu()
            preds = probs.argmax(dim=1)
            rows = _row_dicts(batch_rows)
            for i in range(probs.shape[0]):
                true_index = int(labels[i])
                pred_index = int(preds[i])
                records.append(
                    {
                        "image_path": rows[i]["image_path"],
                        "true_index": true_index,
                        "true_class": BAND_ORDER[true_index],
                        "pred_index": pred_index,
                        "pred_class": BAND_ORDER[pred_index],
                        "prob_pred": float(probs[i, pred_index]),
                        "prob_true": float(probs[i, true_index]),
                        "correct": pred_index == true_index,
                    }
                )
            seen += probs.shape[0]
            if progress_every and step % progress_every == 0:
                rate = seen / max(time.time() - started, 1e-6)
                print(f"[gradcam] scanned {seen} images ({rate:.1f}/s)", flush=True)
    print(f"[gradcam] scan done: {len(records)} images", flush=True)
    return records


def select_examples(
    records: Sequence[Dict[str, Any]],
    per_class: int = 2,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Pick the most confident correct and incorrect example per band.

    "Most confident" is deliberate: a high-confidence mistake is the most
    informative attention pattern to inspect, and a high-confidence match is the
    cleanest reference for what the model looks at when it is right.
    """
    buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "correct": {band: [] for band in BAND_ORDER},
        "incorrect": {band: [] for band in BAND_ORDER},
    }
    for record in records:
        band = record["true_class"]
        group = "correct" if record["correct"] else "incorrect"
        buckets[group][band].append(record)

    picks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"correct": {}, "incorrect": {}}
    for group in ("correct", "incorrect"):
        for band in BAND_ORDER:
            ranked = sorted(
                buckets[group][band], key=lambda r: -float(r["prob_pred"])
            )
            picks[group][band] = ranked[: max(0, int(per_class))]
    return picks


# ---------------------------------------------------------------------------
# Montages
# ---------------------------------------------------------------------------

def _labelled_tile(image: Image.Image, label: str, tile_width: int = 224) -> Image.Image:
    """Return an image with a caption band underneath it."""
    band_height = 34
    canvas = Image.new("RGB", (tile_width, image.height + band_height), (250, 250, 250))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, image.height + 3), label, fill=(20, 20, 20))
    draw.rectangle([(0, 0), (tile_width - 1, canvas.height - 1)], outline=(200, 200, 200))
    return canvas


def write_montage(
    tiles: Sequence[Tuple[Image.Image, str]],
    out_path: Path,
    *,
    columns: int = 2,
) -> Optional[Path]:
    """Write a labelled grid; returns the path, or None when there is nothing to draw."""
    if not tiles:
        return None
    prepared = [_labelled_tile(image, label) for image, label in tiles]
    cols = max(1, min(int(columns), len(prepared)))
    rows = (len(prepared) + cols - 1) // cols
    tile_w = max(t.width for t in prepared)
    tile_h = max(t.height for t in prepared)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (255, 255, 255))
    for i, tile in enumerate(prepared):
        canvas.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def aggregate_attention(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-example attention statistics."""
    if not entries:
        return {"count": 0}
    stats = [e["stats"] for e in entries]
    radii = [s["centroid_radius"] for s in stats]
    return {
        "count": len(entries),
        "fundus_mass": _mean([s["fundus_mass"] for s in stats]),
        "top20_mass": _mean([s["top20_mass"] for s in stats]),
        "centroid_radius": _mean(radii),
        "centroid_radius_min": float(np.min(radii)),
        "centroid_radius_max": float(np.max(radii)),
        "peak_radius": _mean(
            [float(min(1.0, (s["peak_x"] ** 2 + s["peak_y"] ** 2) ** 0.5 / (2.0 ** 0.5))) for s in stats]
        ),
        "mean_confidence": _mean([float(e["prob_pred"]) for e in entries]),
    }


def _band_sentence(band: str, correct: Dict[str, Any], incorrect: Dict[str, Any]) -> str:
    """One paragraph of plain-language reading for a band."""
    parts: List[str] = []
    if correct.get("count"):
        parts.append(
            f"{correct['count']} correctly classified example(s) keep "
            f"{correct['fundus_mass'] * 100:.0f}% of their activation inside the retina and "
            f"put {correct['top20_mass'] * 100:.0f}% of it in the hottest fifth of the frame, "
            f"with the mass centred {correct['centroid_radius']:.2f} (0 = centre, 1 = corner) "
            f"from the middle of the image."
        )
    if incorrect.get("count"):
        parts.append(
            f"{incorrect['count']} misclassified example(s) show "
            f"{incorrect['fundus_mass'] * 100:.0f}% fundus mass, "
            f"{incorrect['top20_mass'] * 100:.0f}% top-fifth concentration and a centroid at "
            f"radius {incorrect['centroid_radius']:.2f}"
        )
        if correct.get("count"):
            delta = incorrect["top20_mass"] - correct["top20_mass"]
            if delta > 0.05:
                parts[-1] += (
                    f" - {delta * 100:.0f} points more concentrated than the correct cases, i.e. "
                    "the wrong prediction commits to fewer, tighter regions."
                )
            elif delta < -0.05:
                parts[-1] += (
                    f" - {abs(delta) * 100:.0f} points less concentrated than the correct cases, "
                    "i.e. the attention is spread out rather than committed to any structure."
                )
            else:
                parts[-1] += " - not meaningfully different in concentration from the correct cases."
        else:
            parts[-1] += "."
    if not parts:
        return f"{band}: no example selected in the scanned subset."
    return f"{band}. " + " ".join(parts)


def build_report(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    compat: Dict[str, Any],
    meta: Dict[str, Any],
    layer_info: Dict[str, Any],
    partition: str,
    rows_scanned: int,
    rows_total: int,
    scan_accuracy: Dict[str, Any],
    entries: Sequence[Dict[str, Any]],
    per_band: Dict[str, Any],
    overlay_root: Path,
    montage_paths: Dict[str, str],
    image_size: int,
    weights_only: bool,
    elapsed: float,
    per_class: int = 2,
) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Grad-CAM Attention Report - Frozen ResNet-50 Baseline")
    add("")
    add("Provisional five-band mapping. Not a clinical result. Grad-CAM is a coarse")
    add("saliency approximation, not lesion segmentation.")
    add("")
    add("## Run under explanation")
    add("")
    add(f"- Run directory: `{run_dir}`")
    add(f"- Checkpoint: `{Path(checkpoint_dir) / 'model.pt'}`")
    add(f"- Checkpoint epoch: {compat.get('checkpoint_epoch')}")
    add(f"- Best epoch (selected on validation): {compat.get('best_epoch')}")
    add(f"- Best validation loss: {compat.get('best_val_loss')}")
    add(
        "- Checkpoint loader: the pipeline's safe-path loader; "
        f"weights_only load accepted = {weights_only} "
        + (
            "(the checkpoint was deserialized with torch.load(weights_only=True))"
            if weights_only
            else "(the embedded torch_version object is not weights_only-safe, so the "
            "pipeline's documented fallback loaded this checkpoint - the same loader the "
            "final test evaluation used)"
        )
    )
    add(f"- Model: frozen ResNet-50 backbone, single trainable 5-way Linear head")
    add(f"- Input resolution: {image_size}x{image_size}, resize-and-pad, ImageNet normalization, no crop")
    add(f"- Partition explained: `{partition}` ({rows_scanned} of {rows_total} rows scanned)")
    add(f"- Wall time: {elapsed:.1f}s")
    add("")
    add("No training, fine-tuning, or checkpoint write happened in this step. The model was")
    add("loaded from disk and evaluated; every number below is inference-only.")
    add("")
    add("## Layer used for Grad-CAM")
    add("")
    add(f"- Module: `{layer_info['module_name']}`")
    add(f"- Type: `{layer_info['module_type']}`")
    add(f"- Output channels: {layer_info['out_channels']}")
    add(f"- Spatial map: {layer_info['spatial_map']}")
    add(f"- Downsample factor: {layer_info['downsample_factor']}x")
    add("")
    add(f"**Why this layer.** {layer_info['justification']}")
    add("")
    add("A 7x7 map upsampled to 224x224 means each cell of the map covers roughly a 32x32")
    add("pixel region of the input. The report therefore talks about *regions* of the")
    add("fundus, never about individual lesions or pixels.")
    add("")
    add("## Scan summary")
    add("")
    add("| Band | Scanned | Correct | Accuracy |")
    add("|---|---|---|---|")
    for band in BAND_ORDER:
        row = scan_accuracy["per_band"].get(band, {})
        add(
            f"| {band} | {row.get('count', 0)} | {row.get('correct', 0)} | "
            f"{row.get('accuracy', 0.0):.4f} |"
        )
    add(
        f"| **All** | {scan_accuracy['count']} | {scan_accuracy['correct']} | "
        f"{scan_accuracy['accuracy']:.4f} |"
    )
    add("")
    add("## Selected examples")
    add("")
    add("| Image | True band | Predicted | P(true) | P(predicted) | Outcome | Overlay |")
    add("|---|---|---|---|---|---|---|")
    for entry in entries:
        outcome = "correct" if entry["correct"] else "incorrect"
        add(
            f"| `{entry['image_path']}` | {entry['true_class']} | {entry['pred_class']} | "
            f"{entry['prob_true']:.4f} | {entry['prob_pred']:.4f} | {outcome} | "
            f"[png]({entry['overlay_rel']}) |"
        )
    add("")
    add("## Attention patterns by band")
    add("")
    add("| Band | Outcome | n | Fundus mass | Top-20% mass | Centroid radius | Mean confidence |")
    add("|---|---|---|---|---|---|---|")
    for band in BAND_ORDER:
        for group in ("correct", "incorrect"):
            agg = per_band[band][group]
            if not agg.get("count"):
                continue
            add(
                f"| {band} | {group} | {agg['count']} | {agg['fundus_mass']:.3f} | "
                f"{agg['top20_mass']:.3f} | {agg['centroid_radius']:.3f} | "
                f"{agg['mean_confidence']:.3f} |"
            )
    add("")
    add("`Fundus mass` is the share of activation inside the retina rather than the black")
    add("pad added by resize-and-pad; values below ~0.9 mean part of the decision is being")
    add("made on padding geometry. `Top-20% mass` is how concentrated the map is (a flat map")
    add("scores 0.2). `Centroid radius` is 0 at image centre and 1 at a corner.")
    add("")
    add("Each row is a mean over the selected examples for that band and outcome - at most")
    add(f"{per_class} of them. These are illustrations of behaviour, not statistics; treat")
    add("differences of a few points as noise, and use the per-image PNGs for the actual")
    add("argument.")
    add("")
    add("### Reading")
    add("")
    add("Each bullet describes the selected images only. With at most a handful per band this")
    add("is an illustration of behaviour, not a finding about why the model errs - do not")
    add("promote these sentences into the paper as results.")
    add("")
    for band in BAND_ORDER:
        add(f"- {_band_sentence(band, per_band[band]['correct'], per_band[band]['incorrect'])}")
    add("")
    add("## Overlays")
    add("")
    add(f"- Per-example PNGs: `{overlay_root}`")
    add(f"  - `correct/<band>/<image>.png` and `incorrect/<band>/<image>.png`")
    add("  - Thin filename convention: `<image>.png` is the CAM for the *predicted* class,")
    add("    which is the class the model actually acted on.")
    for band, path in montage_paths.items():
        add(f"- Montage `{band}`: `{path}`")
    add("")
    add("## Limitations to state alongside these images")
    add("")
    add("1. Grad-CAM is a 7x7 saliency map. It says *where in a coarse grid* the class score")
    add("   is sensitive; it cannot localize microaneurysms, and a hot cell is not a lesion.")
    add("2. The explanation is of the frozen ImageNet backbone's features filtered through a")
    add("   linear head; it is not evidence that the model reasons like a clinician.")
    add("3. The five-band mapping is provisional: Severe NPDR includes Very Severe NPDR and")
    add("   PDR includes Advanced PDR, so band boundaries - and therefore boundary-band")
    add("   explanations - are softer than the labels imply.")
    add("4. Examples were chosen by confidence, not sampled randomly. They are illustrations")
    add("   of behaviour, not a quantitative saliency benchmark.")
    add("5. A proper next step is a deletion/insertion or pointing-game score against expert")
    add("   lesion masks; this report is the qualitative half of that story.")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _prepare(
    project_root: Optional[Path | str],
    dataset_root: Optional[Path | str],
) -> Tuple[Path, Path, Path]:
    """Resolve ``(project_root, split_dir, dataset_root)`` the way the pipeline does."""
    import colab.pipeline as pipe

    root = pipe.discover_project(str(project_root) if project_root else None)
    data_root, checked = sh.resolve_data_root(
        root, explicit=dataset_root or (os_getenv(sh.DATASET_ENV_VAR) or "").strip() or None
    )
    if data_root is None:
        raise FileNotFoundError(
            f"Could not find the fundus corpus. Set {sh.DATASET_ENV_VAR}.\nChecked:\n  - "
            + "\n  - ".join(checked)
        )
    split_dir = root / sh.SPLIT_DIR_NAME
    if not (split_dir / sh.SPLIT_CSV_NAME).is_file():
        raise FileNotFoundError(
            f"Pinned split CSV missing at {split_dir / sh.SPLIT_CSV_NAME}. Restore it; do not "
            "regenerate the audited split."
        )
    return root, split_dir, Path(data_root)


def os_getenv(name: str) -> str:
    """``os.environ.get`` without importing ``os`` at module scope."""
    import os

    return os.environ.get(name, "")


def discover_run_dir(root: Path, explicit: Optional[Path | str] = None) -> Path:
    """Return the run directory holding a best checkpoint."""
    if explicit:
        out = Path(explicit)
        if not (out / train_mod.BEST_CHECKPOINT_DIR_NAME / "model.pt").is_file():
            raise FileNotFoundError(
                f"{out} does not look like a completed run: no "
                f"{train_mod.BEST_CHECKPOINT_DIR_NAME}/model.pt inside it."
            )
        return out

    candidates = [
        d
        for d in (root / "cpu_runs").glob("*")
        if (d / train_mod.BEST_CHECKPOINT_DIR_NAME / "model.pt").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No run with a best checkpoint found under cpu_runs/. Pass --output-dir explicitly."
        )
    newest = max(candidates, key=lambda d: (d / train_mod.BEST_CHECKPOINT_DIR_NAME / "model.pt").stat().st_mtime)
    print(f"[gradcam] no --output-dir given; using most recent run {newest}", flush=True)
    return newest


def _require_completed(out: Path) -> Optional[str]:
    """Return the run status, refusing a run that never completed."""
    try:
        import colab.pipeline as pipe

        status = pipe.run_state(out)["status"]
    except Exception:
        return None
    if status != "completed":
        raise RuntimeError(
            f"Refusing to explain the run at {out}: it is {status!r}, not 'completed'. "
            "Explanation runs against the checkpoint that the final evaluation used."
        )
    return status


def run_gradcam(
    *,
    project_root: Optional[Path | str] = None,
    dataset_root: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    checkpoint_dir: Optional[Path | str] = None,
    partition: str = "validation",
    per_class: int = 2,
    max_scan: int = 400,
    image_size: Optional[int] = None,
    batch_size: int = 32,
    seed: int = 0,
    device: Optional[torch.device] = None,
    alpha: float = 0.45,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Generate Grad-CAM overlays and an attention report for the saved checkpoint.

    Returns a summary dict; writes overlays and both reports under
    ``<run>/explainability/gradcam``.
    """
    if partition not in PARTITION_CHOICES:
        raise ValueError(f"partition must be one of {PARTITION_CHOICES}, got {partition!r}.")
    if partition == "test" and not confirm:
        raise PermissionError(
            "Explaining the held-out test partition is guarded. Re-run with --confirm."
        )

    started = time.time()
    root, split_dir, data_root = _prepare(project_root, dataset_root)
    out = discover_run_dir(root, output_dir)
    _require_completed(out)

    best_dir = Path(checkpoint_dir) if checkpoint_dir else (out / train_mod.BEST_CHECKPOINT_DIR_NAME)
    model_path = best_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint weights missing: {model_path}")

    compat = train_mod.checkpoint_compat_report(best_dir)
    meta = train_mod.inspect_checkpoint_metadata(best_dir)
    config = meta.get("config", {}) or {}
    if image_size is None:
        image_size = int(config.get("image_size") or 224)
    image_size = int(image_size)

    if device is None:
        device = train_mod.resolve_device()
    device = torch.device(device)

    # Architecture with weights=None: evaluation/explanation restores trained
    # weights, so the ImageNet download is neither needed nor triggered here.
    model = build_frozen_resnet50(num_classes=len(BAND_ORDER), pretrained=False)
    payload, weights_only = train_mod._torch_load(model_path, device)
    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint {model_path} has no 'model_state_dict' entry.")
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    rows = filter_by_partition(load_split_csv(split_dir), partition)
    if not rows:
        raise ValueError(f"Split CSV contains no {partition!r} rows.")

    # Bounded, deterministic scan: shuffle with the given seed so the scanned slice
    # is class-diverse, then keep at most max_scan rows.
    shuffled = list(rows)
    random.Random(int(seed)).shuffle(shuffled)
    scan_rows = shuffled[: max(1, min(int(max_scan), len(shuffled)))]
    dataset = train_mod.PrimarySplitDataset(scan_rows, data_root, image_size=image_size)
    records = scan_partition(model, dataset, device, batch_size=batch_size)

    scan_correct = sum(1 for r in records if r["correct"])
    per_band_scan: Dict[str, Any] = {}
    for band in BAND_ORDER:
        band_records = [r for r in records if r["true_class"] == band]
        hits = sum(1 for r in band_records if r["correct"])
        per_band_scan[band] = {
            "count": len(band_records),
            "correct": hits,
            "accuracy": (hits / len(band_records)) if band_records else 0.0,
        }
    scan_accuracy = {
        "count": len(records),
        "correct": scan_correct,
        "accuracy": (scan_correct / len(records)) if records else 0.0,
        "per_band": per_band_scan,
    }

    picks = select_examples(records, per_class=per_class)
    layer_name, target_layer = resolve_cam_layer(model)
    layer_info = layer_facts(model, layer_name, target_layer, image_size)
    cam_engine = GradCAM(model, target_layer, device=device)

    overlay_root = out / EXPLAIN_DIR_NAME / GRADCAM_DIR_NAME / partition
    entries: List[Dict[str, Any]] = []
    montage_tiles: Dict[str, List[Tuple[Image.Image, str]]] = {band: [] for band in BAND_ORDER}

    for band in BAND_ORDER:
        for group in ("correct", "incorrect"):
            for record in picks[group][band]:
                image_path = Path(record["image_path"])
                full_path = data_root / image_path
                tensor, base = load_model_input(full_path, image_size)
                class_index = int(record["pred_index"])
                cam = cam_engine(tensor.unsqueeze(0), [class_index])[0]
                base_rgb = np.asarray(base.convert("RGB"), dtype=np.uint8)
                overlay = render_overlay(base_rgb, cam, alpha=alpha)

                slug = _slug(image_path.stem)
                rel_dir = Path(group) / _slug(band)
                out_path = overlay_root / rel_dir / f"{slug}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                overlay.save(out_path)

                stats = attention_stats(cam, fundus_mask(base_rgb))
                entry = {
                    "image_path": record["image_path"],
                    "true_class": record["true_class"],
                    "pred_class": record["pred_class"],
                    "correct": bool(record["correct"]),
                    "prob_true": float(record["prob_true"]),
                    "prob_pred": float(record["prob_pred"]),
                    "cammed_class": record["pred_class"],
                    "overlay_rel": str(out_path.relative_to(out)).replace("\\", "/"),
                    "stats": stats,
                }
                entries.append(entry)

                label = f"{image_path.stem}\n{record['true_class']} -> {record['pred_class']}\n{group}"
                montage_tiles[band].append((overlay, label))

    montage_paths: Dict[str, str] = {}
    for band, tiles in montage_tiles.items():
        if not tiles:
            continue
        path = write_montage(
            tiles, overlay_root / f"montage_{_slug(band)}.png", columns=2
        )
        if path is not None:
            montage_paths[band] = str(path.relative_to(out)).replace("\\", "/")

    per_band: Dict[str, Dict[str, Any]] = {}
    for band in BAND_ORDER:
        per_band[band] = {}
        for group in ("correct", "incorrect"):
            group_entries = [
                e for e in entries if e["true_class"] == band and e["correct"] == (group == "correct")
            ]
            per_band[band][group] = aggregate_attention(group_entries)

    elapsed = time.time() - started
    report_md = build_report(
        run_dir=out,
        checkpoint_dir=best_dir,
        compat=compat,
        meta=meta,
        layer_info=layer_info,
        partition=partition,
        rows_scanned=len(scan_rows),
        rows_total=len(rows),
        scan_accuracy=scan_accuracy,
        entries=entries,
        per_band=per_band,
        overlay_root=overlay_root.relative_to(out).as_posix(),
        montage_paths=montage_paths,
        image_size=image_size,
        weights_only=bool(weights_only),
        elapsed=elapsed,
        per_class=per_class,
    )

    explain_dir = out / EXPLAIN_DIR_NAME / GRADCAM_DIR_NAME
    explain_dir.mkdir(parents=True, exist_ok=True)
    md_path = explain_dir / REPORT_MD_NAME
    md_path.write_text(report_md, encoding="utf-8")

    payload_summary = {
        "checkpoint": str(model_path),
        "checkpoint_epoch": compat.get("checkpoint_epoch"),
        "best_epoch": compat.get("best_epoch"),
        "best_val_loss": compat.get("best_val_loss"),
        "weights_only_load": bool(weights_only),
        "partition": partition,
        "image_size": image_size,
        "device": str(device),
        "rows_scanned": len(scan_rows),
        "rows_in_partition": len(rows),
        "scan_accuracy": scan_accuracy,
        "cam_layer": layer_info,
        "per_class": per_class,
        "max_scan": max_scan,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "examples": entries,
        "per_band_attention": per_band,
        "montages": montage_paths,
        "overlay_root": str(overlay_root),
        "report_markdown": str(md_path),
        "provisional_note": train_mod.PROVISIONAL_NOTE,
        "retrained": False,
    }
    json_path = explain_dir / REPORT_JSON_NAME
    json_path.write_text(
        json.dumps(payload_summary, indent=2, default=str), encoding="utf-8"
    )

    return {
        "ok": True,
        "command": "gradcam",
        "output_dir": str(out),
        "checkpoint": str(model_path),
        "cam_layer": layer_info["module_name"],
        "partition": partition,
        "examples": len(entries),
        "correct_examples": sum(1 for e in entries if e["correct"]),
        "incorrect_examples": sum(1 for e in entries if not e["correct"]),
        "report_markdown": str(md_path),
        "report_json": str(json_path),
        "overlay_root": str(overlay_root),
        "montages": montage_paths,
        "scan_accuracy": round(scan_accuracy["accuracy"], 4),
        "seconds": round(elapsed, 1),
        "retrained": False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m colab.gradcam",
        description=(
            "Generate Grad-CAM overlays and an attention report for an existing "
            "frozen-ResNet-50 baseline checkpoint. Inference only: nothing is trained, "
            "and no checkpoint, dataset, or split artifact is modified."
        ),
    )
    parser.add_argument("--project-root", default=None, help="Project root (contains dataset/ and colab/).")
    parser.add_argument("--dataset-root", default=None, help=f"Fundus corpus root (or set {sh.DATASET_ENV_VAR}).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory holding best_checkpoint/. Default: newest run under cpu_runs/.",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="Explicit checkpoint directory override.")
    parser.add_argument(
        "--partition",
        choices=list(PARTITION_CHOICES),
        default="validation",
        help="Partition to explain (default: validation).",
    )
    parser.add_argument("--confirm", action="store_true", help="Required to explain the test partition.")
    parser.add_argument("--per-class", type=int, default=2, help="Examples per band and outcome (default: 2).")
    parser.add_argument("--max-scan", type=int, default=400, help="Maximum rows to scan for examples (default: 400).")
    parser.add_argument("--image-size", type=int, default=None, help="Input size; default: from checkpoint config.")
    parser.add_argument("--batch-size", type=int, default=32, help="Scan batch size (default: 32).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the deterministic scan slice (default: 0).")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:0. Default: auto-detect.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay blend strength (default: 0.45).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_gradcam(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            partition=args.partition,
            per_class=args.per_class,
            max_scan=args.max_scan,
            image_size=args.image_size,
            batch_size=args.batch_size,
            seed=args.seed,
            device=torch.device(args.device) if args.device else None,
            alpha=args.alpha,
            confirm=args.confirm,
        )
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"[gradcam] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
