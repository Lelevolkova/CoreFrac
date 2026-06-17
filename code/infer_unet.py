"""Distilled U-Net inference: folder of core images -> binary crack masks.

Pure frozen inference (no GT, no SAM 3, no LLM). For each image we cut 2:1 vertical
tiles (height = 2x width, the deployment geometry), run the distilled U-Net per tile,
and OR-merge predictions back to the full-resolution strip. Masks are written as
PNG (255 = crack, 0 = background) named ``<stem>_binary_mask.png``.

Usage:
  python infer_unet.py --images <dir> --out <dir> --ckpt <unet_distilled.pt>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE))
os.environ.setdefault("COREFRAC_ROOT", str(BUNDLE))  # harmless; module import only

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def norm_stem(p: Path) -> str:
    s = p.stem
    return s[:-9] if s.endswith("_original") else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tile-aspect", type=float, default=2.0)
    args = ap.parse_args()

    import torch

    from problems.prompts.corefrac.utils.dataset import (
        iter_vertical_windows,
    )
    from distill_unet import _build_unet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    net_size = int(ckpt.get("net_size", 256))
    net = _build_unet(ckpt["width"], ckpt["depth"])
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    print(f"[infer] U-Net {ckpt.get('n_params', '?')} params, net_size={net_size}, "
          f"thr={args.threshold}, device={device}", flush=True)

    def unet_predict(crop_rgb: Image.Image, out_hw: tuple) -> np.ndarray:
        arr = np.asarray(crop_rgb.resize((net_size, net_size), Image.Resampling.BILINEAR)).astype(np.float32)
        arr = (arr / 255.0 - IM_MEAN) / IM_STD
        x = torch.from_numpy(np.transpose(arr, (2, 0, 1))[None]).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(net(x))[0, 0].cpu().numpy()
        hard = (prob > args.threshold).astype(np.uint8) * 255
        im = Image.fromarray(hard).resize((out_hw[1], out_hw[0]), Image.Resampling.NEAREST)
        return np.asarray(im) > 127

    def windows_for(H: int, W: int):
        tile_h = max(1, min(H, int(round(args.tile_aspect * W))))
        overlap = 0 if tile_h >= H else min(W // 2, tile_h - 1)
        return iter_vertical_windows(np.zeros((H, W), dtype=bool), height=tile_h, overlap=max(0, overlap))

    images = sorted(p for p in Path(args.images).iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXT)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[infer] {len(images)} images -> {out_dir}", flush=True)

    t0 = time.time()
    for i, img_path in enumerate(images):
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            W, H = rgb.width, rgb.height
            full = np.asarray(rgb)
        wins = windows_for(H, W)
        mask = np.zeros((H, W), dtype=bool)
        for win in wins:
            y0, y1 = win.y0, win.y1
            crop = Image.fromarray(full[y0:y1, :])
            mask[y0:y1, :] |= unet_predict(crop, (y1 - y0, W))
        out_path = out_dir / f"{norm_stem(img_path)}_binary_mask.png"
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out_path)
        cov = 100.0 * mask.mean()
        print(f"[infer] {i + 1}/{len(images)} {img_path.name} {W}x{H} "
              f"tiles={len(wins)} crack={cov:.2f}% -> {out_path.name}", flush=True)

    print(f"[infer] DONE {len(images)} masks in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
