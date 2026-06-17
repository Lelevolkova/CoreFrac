"""Lazy SAM3 image inference wrapper for geological fracture segmentation."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

from problems.prompts.corefrac.config import (
    SAM3_CONFIG,
    SAM3_REPO,
)

# When this tool runs inside the evolution engine, the worker subprocess uses
# fd 1 (stdout) as a length-prefixed IPC channel back to the parent. transformers
# weight-loading progress bars write to fd 1 at the native level, bypassing
# Python-level stdout redirection, and corrupt that protocol (parent then hangs
# until stage_timeout). Disable HF/transformers progress output up front, and
# additionally redirect fd 1 -> fd 2 at the OS level while building the runtime.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_RUNTIME: Any | None = None
_SAM3_IMPORT_ERROR: str | None = None


@contextlib.contextmanager
def _redirect_fd1_to_fd2():
    """Send anything written to fd 1 to fd 2 for the duration of the block.

    Guards the worker's stdout IPC protocol against native-level prints (e.g.
    transformers weight-loading progress) emitted while loading the model.
    """
    try:
        saved_fd1 = os.dup(1)
    except OSError:
        # No real fd 1 (rare); nothing to protect.
        yield
        return
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(saved_fd1, 1)
        os.close(saved_fd1)


@dataclass(frozen=True)
class _TransformersRuntime:
    model: Any
    processor: Any
    device: str
    confidence_threshold: float


def _select_cuda_index(n_gpus: int) -> int:
    """Pick a CUDA device index, stable for the lifetime of this process.

    Concurrent validation workers each run in their own process; a file-lock
    round-robin counter spreads them evenly across all visible GPUs so a single
    GPU is never oversubscribed (which would cause OOM / stage timeouts).
    An explicit ``SAM3_CUDA_INDEX`` override wins when set.
    """
    explicit = os.environ.get("SAM3_CUDA_INDEX", "").strip()
    if explicit:
        return max(0, min(n_gpus - 1, int(explicit)))

    import fcntl

    slot_dir = Path(os.environ.get("SAM3_GPU_SLOT_DIR", "/tmp/corefrac_gpu_slots"))
    slot_dir.mkdir(parents=True, exist_ok=True)
    counter = slot_dir / "counter"
    with open(counter, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            raw = fh.read().strip()
            value = int(raw) if raw else 0
            fh.seek(0)
            fh.truncate()
            fh.write(str(value + 1))
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return value % n_gpus


def _resolve_device() -> str:
    import torch

    device = SAM3_CONFIG["device"]
    if device == "cuda":
        if not torch.cuda.is_available():
            if SAM3_CONFIG.get("allow_cpu", False):
                return "cpu"
            raise RuntimeError(
                "SAM3_DEVICE=cuda was requested, but CUDA is not available. "
                "Set SAM3_ALLOW_CPU=true only for local debugging."
            )
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            return f"cuda:{_select_cuda_index(n_gpus)}"
    return device


def _cuda_dtype(device: str):
    import torch

    if device == "cpu":
        return torch.float32
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    major, _minor = torch.cuda.get_device_capability(index)
    return torch.bfloat16 if major >= 8 else torch.float16


def _ensure_native_sam3_importable() -> None:
    global _SAM3_IMPORT_ERROR
    if _SAM3_IMPORT_ERROR is not None:
        raise ImportError(_SAM3_IMPORT_ERROR)

    repo = SAM3_REPO.resolve()
    repo_str = str(repo)
    if repo.is_dir() and repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        import sam3  # noqa: F401
    except ImportError as exc:
        _SAM3_IMPORT_ERROR = (
            f"SAM3 is not importable from {repo}. "
            f"Install with `pip install -e {repo}` or set SAM3_REPO."
        )
        raise ImportError(_SAM3_IMPORT_ERROR) from exc


def _load_transformers_runtime() -> _TransformersRuntime:
    from transformers import Sam3Model, Sam3Processor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    with contextlib.suppress(Exception):
        hf_logging.disable_progress_bar()

    device = _resolve_device()
    model_id = SAM3_CONFIG["model_name"]
    dtype = _cuda_dtype(device)

    # fd 1 -> fd 2 guard: weight-loading progress must not reach the IPC pipe.
    with _redirect_fd1_to_fd2():
        model = Sam3Model.from_pretrained(model_id, torch_dtype=dtype)
        model.eval()
        if device != "cpu":
            model = model.to(device)

        processor = Sam3Processor.from_pretrained(model_id)
    return _TransformersRuntime(
        model=model,
        processor=processor,
        device=device,
        confidence_threshold=SAM3_CONFIG["confidence_threshold"],
    )


def _load_native_runtime() -> Any:
    _ensure_native_sam3_importable()

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = _resolve_device()
    checkpoint_path = SAM3_CONFIG.get("checkpoint_path")
    load_from_hf = SAM3_CONFIG.get("load_from_hf", False)

    build_kwargs: dict[str, Any] = {"device": device, "load_from_HF": load_from_hf}
    if checkpoint_path:
        build_kwargs["checkpoint_path"] = checkpoint_path

    model = build_sam3_image_model(**build_kwargs)
    model = model.to(device)
    if device == "cpu":
        model = model.float()
    model.eval()

    return Sam3Processor(
        model,
        resolution=SAM3_CONFIG["resolution"],
        device=device,
        confidence_threshold=SAM3_CONFIG["confidence_threshold"],
    )


def _get_runtime() -> Any:
    global _RUNTIME, _SAM3_IMPORT_ERROR
    if _RUNTIME is not None:
        return _RUNTIME

    backend = SAM3_CONFIG.get("backend", "transformers")
    if backend == "native":
        _RUNTIME = _load_native_runtime()
    elif backend == "transformers":
        _RUNTIME = _load_transformers_runtime()
    else:
        raise ValueError(f"Unsupported SAM3 backend: {backend}")
    return _RUNTIME


def _resize_bool_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    if mask.shape == (height, width):
        return mask.astype(bool)

    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.array(resized) > 127


def _union_mask_tensor(masks: Any, threshold: float) -> tuple[np.ndarray, int]:
    import torch

    if masks is None:
        return np.zeros((1, 1), dtype=bool), 0

    if isinstance(masks, torch.Tensor):
        mask_np = masks.detach().cpu().numpy()
    else:
        mask_np = np.asarray(masks)

    if mask_np.ndim == 4:
        mask_np = mask_np[:, 0, :, :]
    elif mask_np.ndim == 2:
        mask_np = mask_np[None, :, :]
    elif mask_np.ndim != 3:
        raise ValueError(f"Unexpected SAM3 mask shape: {mask_np.shape}")

    n_instances = int(mask_np.shape[0])
    if n_instances == 0:
        return np.zeros(mask_np.shape[-2:], dtype=bool), 0

    union = np.any(mask_np > threshold, axis=0)
    return union.astype(bool), n_instances


def _segment_transformers(
    image: Image.Image, text_prompt: str
) -> tuple[np.ndarray, int]:
    import torch

    runtime = _get_runtime()
    assert isinstance(runtime, _TransformersRuntime)

    threshold = runtime.confidence_threshold
    inputs = runtime.processor(
        images=image, text=text_prompt.strip(), return_tensors="pt"
    )
    if runtime.device != "cpu":
        inputs = {
            key: value.to(runtime.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    with torch.no_grad(), _redirect_fd1_to_fd2():
        outputs = runtime.model(**inputs)

    results = runtime.processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return _union_mask_tensor(results.get("masks"), threshold)


def _segment_native(image: Image.Image, text_prompt: str) -> tuple[np.ndarray, int]:
    processor = _get_runtime()
    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(
        prompt=text_prompt.strip(), state=inference_state
    )
    return _union_mask_tensor(inference_state.get("masks"), 0.5)


def sam3_segment_cracks(image_path: str, text_prompt: str) -> dict[str, Any]:
    """Run SAM3 text-prompt segmentation and return union binary mask."""
    started = time.perf_counter()
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        target_shape = (image.height, image.width)
        backend = SAM3_CONFIG.get("backend", "transformers")
        if backend == "native":
            union, n_instances = _segment_native(image, text_prompt)
        else:
            union, n_instances = _segment_transformers(image, text_prompt)

        union = _resize_bool_mask(union, target_shape)

        return {
            "pred_mask": union.astype(bool).tolist(),
            "n_instances": n_instances,
            "latency_sec": time.perf_counter() - started,
            "success": n_instances > 0,
            "error": None,
        }
    except Exception as exc:
        return {
            "pred_mask": np.zeros((1, 1), dtype=bool).tolist(),
            "n_instances": 0,
            "latency_sec": time.perf_counter() - started,
            "success": False,
            "error": str(exc),
        }


def reset_sam3_runtime() -> None:
    """Reset lazy SAM3 singleton (for tests)."""
    global _RUNTIME, _SAM3_IMPORT_ERROR
    _RUNTIME = None
    _SAM3_IMPORT_ERROR = None
