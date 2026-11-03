"""Hardware, operating-system and dependency detection.

The pipeline must behave sensibly on somebody else's Windows machine. This
module answers, without assuming anything:

* which OS / Python / architecture is running,
* whether an NVIDIA GPU is physically present,
* whether the installed PyTorch build can actually *use* CUDA,
* how much RAM and disk are available,
* which batch size and worker count are safe,
* whether mixed precision will help.

A present NVIDIA GPU is deliberately **not** treated as proof that CUDA works.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HardwareReport:
    """Everything the pipeline needs to know about the machine."""

    os_name: str
    os_version: str
    os_release: str
    architecture: str
    python_version: str
    python_executable: str
    is_64bit: bool
    cpu_count: int
    ram_total_gb: Optional[float]
    ram_available_gb: Optional[float]
    disk_free_gb: Optional[float]
    nvidia_smi_present: bool
    nvidia_gpu_names: List[str] = field(default_factory=list)
    nvidia_driver_version: Optional[str] = None
    torch_installed: bool = False
    torch_version: Optional[str] = None
    torch_cuda_build: Optional[str] = None
    torchvision_version: Optional[str] = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_names: List[str] = field(default_factory=list)
    cudnn_version: Optional[str] = None
    compute_capabilities: List[str] = field(default_factory=list)
    selected_device: str = "cpu"
    mixed_precision: bool = False
    recommended_classification_batch_size: int = 8
    recommended_segmentation_batch_size: int = 2
    recommended_num_workers: int = 0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def gpu_present_but_unusable(self) -> bool:
        """True when an NVIDIA GPU exists but PyTorch cannot use it."""
        return self.nvidia_smi_present and not self.cuda_available

    def summary_lines(self) -> List[str]:
        lines = [
            f"OS                 : {self.os_name} {self.os_release} ({self.os_version}) {self.architecture}",
            f"Python             : {self.python_version}  [{self.python_executable}]",
            f"CPU cores          : {self.cpu_count}",
            f"RAM                : {self.ram_total_gb or '?'} GB total, "
            f"{self.ram_available_gb or '?'} GB available",
            f"Disk free (root)   : {self.disk_free_gb if self.disk_free_gb is not None else '?'} GB",
            f"NVIDIA GPU present : {self.nvidia_smi_present}"
            + (f" ({', '.join(self.nvidia_gpu_names)})" if self.nvidia_gpu_names else ""),
            f"PyTorch            : {self.torch_version or 'not installed'}"
            + (f" (CUDA build {self.torch_cuda_build})" if self.torch_cuda_build else " (CPU build)"),
            f"torchvision        : {self.torchvision_version or 'not installed'}",
            f"CUDA usable        : {self.cuda_available}"
            + (f" ({', '.join(self.cuda_device_names)})" if self.cuda_device_names else ""),
            f"Selected device    : {self.selected_device}",
            f"Mixed precision    : {'enabled (AMP)' if self.mixed_precision else 'disabled (fp32)'}",
            f"Batch sizes        : classification={self.recommended_classification_batch_size}, "
            f"segmentation={self.recommended_segmentation_batch_size}",
            f"DataLoader workers : {self.recommended_num_workers}",
        ]
        for note in self.notes:
            lines.append(f"NOTE               : {note}")
        return lines


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _windows_version() -> tuple[str, str, str]:
    if os.name != "nt":
        return platform.system(), platform.release(), platform.version()
    name = "Windows"
    release = platform.release()
    version = platform.version()
    build = ""
    try:  # Prefer the real build number from the OS itself.
        out = subprocess.run(
            ["cmd", "/c", "ver"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        if out:
            build = out
    except Exception:
        pass
    return name, f"{release} {build}".strip(), version


def _total_ram_gb() -> tuple[Optional[float], Optional[float]]:
    try:
        import psutil  # optional dependency

        vm = psutil.virtual_memory()
        return round(vm.total / 1024**3, 2), round(vm.available / 1024**3, 2)
    except Exception:
        pass
    if os.name == "nt":
        try:
            out = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_OperatingSystem) | "
                    "Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json",
                ],
                capture_output=True, text=True, timeout=20,
            ).stdout
            import json

            data = json.loads(out)
            total = float(data.get("TotalVisibleMemorySize", 0)) / 1024**2  # KiB -> GB
            free = float(data.get("FreePhysicalMemory", 0)) / 1024**2
            if total > 0:
                return round(total, 2), round(free, 2)
        except Exception:
            pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page / 1024**3
        return round(total, 2), None
    except Exception:
        return None, None


def _nvidia_smi() -> tuple[bool, List[str], Optional[str]]:
    """Query nvidia-smi directly: hardware presence, independent of PyTorch."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, [], None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode != 0:
            return False, [], None
        names: List[str] = []
        driver: Optional[str] = None
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0]:
                names.append(parts[0])
            if len(parts) > 1 and parts[1]:
                driver = parts[1]
        return bool(names), names, driver
    except Exception:
        return False, [], None


def _torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_build": None,
        "torchvision": None,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
        "cudnn": None,
        "capabilities": [],
    }
    try:
        import torch
    except Exception:
        return info
    info["installed"] = True
    info["version"] = getattr(torch, "__version__", None)
    info["cuda_build"] = getattr(torch.version, "cuda", None)
    try:
        import torchvision

        info["torchvision"] = getattr(torchvision, "__version__", None)
    except Exception:
        pass
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["cuda_available"] = False
    if info["cuda_available"]:
        try:
            info["device_count"] = int(torch.cuda.device_count())
            info["device_names"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
            info["capabilities"] = [
                f"{major}.{minor}"
                for major, minor in (torch.cuda.get_device_capability(i) for i in range(info["device_count"]))
            ]
        except Exception:
            pass
        try:
            import torch.backends.cudnn as cudnn

            if cudnn.is_available():
                info["cudnn"] = str(cudnn.version())
        except Exception:
            pass
    return info


def _disk_free_gb(path: Path) -> Optional[float]:
    try:
        target = path
        while not target.exists() and target.parent != target:
            target = target.parent
        return round(shutil.disk_usage(str(target)).free / 1024**3, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sizing heuristics
# ---------------------------------------------------------------------------

def _tiered_batch_size(
    ram_available_gb: Optional[float],
    vram_gb: Optional[float],
    *,
    on_gpu: bool,
    tiers: List[tuple[float, int]],
    floor: int,
    ceil: int,
) -> int:
    budget = vram_gb if on_gpu else ram_available_gb
    if budget is None:
        return max(floor, min(ceil, floor))
    for threshold, size in tiers:
        if budget >= threshold:
            return max(floor, min(ceil, size))
    return floor


def _vram_gb(device_index: int = 0) -> Optional[float]:
    try:
        import torch

        props = torch.cuda.get_device_properties(device_index)
        return round(props.total_memory / 1024**3, 2)
    except Exception:
        return None


def detect_hardware(
    *,
    mixed_precision_preference: str = "auto",
    disk_probe: Optional[Path] = None,
) -> HardwareReport:
    """Build a :class:`HardwareReport` for this machine."""
    name, release, version = _windows_version()
    ram_total, ram_available = _total_ram_gb()
    smi_present, gpu_names, driver = _nvidia_smi()
    torch_info = _torch_info()

    report = HardwareReport(
        os_name=name,
        os_version=version,
        os_release=release,
        architecture=platform.machine(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        is_64bit=sys.maxsize > 2**32,
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        disk_free_gb=_disk_free_gb(disk_probe or Path.cwd()),
        nvidia_smi_present=smi_present,
        nvidia_gpu_names=gpu_names,
        nvidia_driver_version=driver,
        torch_installed=torch_info["installed"],
        torch_version=torch_info["version"],
        torch_cuda_build=torch_info["cuda_build"],
        torchvision_version=torch_info["torchvision"],
        cuda_available=torch_info["cuda_available"],
        cuda_device_count=torch_info["device_count"],
        cuda_device_names=torch_info["device_names"],
        cudnn_version=torch_info["cudnn"],
        compute_capabilities=torch_info["capabilities"],
    )

    on_gpu = report.cuda_available and report.cuda_device_count > 0
    report.selected_device = "cuda:0" if on_gpu else "cpu"

    # Mixed precision: AMP on CUDA is a real speed/VRAM win. On CPU the bf16 path
    # is not reliably faster and fp16 is unsupported broadly, so it stays off.
    pref = str(mixed_precision_preference).strip().lower()
    if pref in {"true", "yes", "on", "1"}:
        report.mixed_precision = on_gpu
        if not on_gpu:
            report.notes.append("mixed precision was requested but no usable CUDA device exists; staying fp32")
    elif pref in {"false", "no", "off", "0"}:
        report.mixed_precision = False
    else:
        report.mixed_precision = on_gpu

    vram = _vram_gb(0) if on_gpu else None
    report.recommended_classification_batch_size = _tiered_batch_size(
        ram_available,
        vram,
        on_gpu=on_gpu,
        tiers=[(20.0, 64), (11.0, 48), (7.0, 32), (3.0, 16)],
        floor=8,
        ceil=64,
    )
    report.recommended_segmentation_batch_size = _tiered_batch_size(
        ram_available,
        vram,
        on_gpu=on_gpu,
        tiers=[(20.0, 16), (11.0, 8), (7.0, 6), (3.0, 4)],
        floor=2,
        ceil=16,
    )
    # Windows DataLoader workers with spawn semantics cost more than they save
    # for a dataset this small per batch, so default to 0 there.
    if on_gpu and os.name != "nt":
        report.recommended_num_workers = min(8, max(2, report.cpu_count // 2))
    elif on_gpu:
        report.recommended_num_workers = min(4, max(2, report.cpu_count // 2))
    else:
        report.recommended_num_workers = 0

    if not report.torch_installed:
        report.notes.append("PyTorch is not importable; run scripts/setup_env.py before training")
    elif report.torch_cuda_build is None and smi_present:
        report.notes.append(
            "an NVIDIA GPU is present but the installed PyTorch is a CPU-only build; "
            "reinstall a CUDA build to use it"
        )
    elif report.gpu_present_but_unusable:
        report.notes.append(
            "nvidia-smi reports a GPU but torch.cuda.is_available() is False "
            "(driver/CUDA mismatch); training will use the CPU"
        )
    if report.disk_free_gb is not None and report.disk_free_gb < 40:
        report.notes.append(
            f"only {report.disk_free_gb} GB free on the project drive; MMRDR needs ~19 GB "
            "and IDRiD ~2 GB"
        )
    if not report.is_64bit:
        report.notes.append("32-bit Python detected; PyTorch wheels require 64-bit Python")

    return report


def configure_torch_threads(cpu_count: Optional[int] = None) -> int:
    """Bound intra-op threads so CPU training does not thrash."""
    try:
        import torch

        count = cpu_count or max(1, (os.cpu_count() or 4) - 1)
        torch.set_num_threads(count)
        return count
    except Exception:
        return 0


def resolve_device(preference: str = "auto"):
    """Return a ``torch.device`` from a preference string."""
    import torch

    pref = (preference or "auto").strip().lower()
    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(pref)
        return torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def dependency_report() -> Dict[str, Any]:
    """Report which optional/required packages are importable."""
    modules = {
        "torch": "PyTorch",
        "torchvision": "torchvision",
        "numpy": "NumPy",
        "pandas": "pandas",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "yaml": "PyYAML",
        "scipy": "SciPy",
        "tqdm": "tqdm",
    }
    out: Dict[str, Any] = {}
    for module, label in modules.items():
        try:
            imported = __import__(module)
            out[label] = {"available": True, "version": getattr(imported, "__version__", "unknown")}
        except Exception as exc:
            out[label] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return out
