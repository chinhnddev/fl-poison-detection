from __future__ import annotations

from typing import Optional


def _normalize_device_string(device: Optional[str]) -> str:
    raw = "cpu" if device is None else str(device).strip()
    return raw or "cpu"


def normalize_ultralytics_device(device: Optional[str]) -> str:
    """Normalize common torch-style device strings for Ultralytics APIs."""
    raw = _normalize_device_string(device)
    lower = raw.lower()

    if lower.startswith("cuda:"):
        suffix = raw.split(":", 1)[1].strip()
        return suffix or "0"
    if lower == "cuda":
        return "0"
    return raw


def resolve_eval_device(device: Optional[str]) -> str:
    """Return a validated device string and fail loudly on unavailable CUDA."""
    normalized = normalize_ultralytics_device(device)
    lower = normalized.lower()

    if lower in {"cpu", "mps"}:
        return normalized
    if lower.startswith("npu") or lower.startswith("intel") or lower.startswith("vulkan"):
        return normalized

    wants_cuda = False
    if lower.isdigit():
        wants_cuda = True
    elif all(part.strip().isdigit() for part in lower.split(",")):
        wants_cuda = True

    if not wants_cuda:
        return normalized

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            f"Requested CUDA device '{device}', but PyTorch could not be imported to validate CUDA availability. "
            f"Original import error: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device '{device}', but torch.cuda.is_available() is False. "
            "This environment is falling back to CPU because CUDA is not available to PyTorch."
        )

    requested = [int(part.strip()) for part in lower.split(",") if part.strip()]
    if requested:
        max_index = torch.cuda.device_count() - 1
        missing = [idx for idx in requested if idx > max_index]
        if missing:
            raise RuntimeError(
                f"Requested CUDA device '{device}', but only {torch.cuda.device_count()} CUDA device(s) are visible "
                f"to PyTorch."
            )

    return normalized
