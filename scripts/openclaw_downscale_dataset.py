#!/usr/bin/env python3
"""Downscale dataset images to a target ratio.

Default behavior matches Silas' current workflow:
- Input:  ~/Templates/DJI-Mini3-Pro/20260208/102MEDIA/DJI_0544/images
- Output: data/DJI_0544_0.5/images
- Scale:  0.5
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_src = Path.home() / "Templates/DJI-Mini3-Pro/20260208/102MEDIA/DJI_0544/images"
    default_dst = repo_root / "data/DJI_0544_0.5/images"

    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=default_src, help="source image directory")
    p.add_argument("--dst", type=Path, default=default_dst, help="output image directory")
    p.add_argument("--scale", type=float, default=0.5, help="resize ratio, e.g. 0.5")
    p.add_argument("--workers", type=int, default=min(12, max(1, os.cpu_count() or 4)))
    p.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    return p.parse_args()


def resize_one(src_path: Path, dst_path: Path, scale: float, overwrite: bool) -> str:
    if dst_path.exists() and not overwrite:
        return "skip"

    with Image.open(src_path) as im:
        w, h = im.size
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = im.resize((nw, nh), Image.Resampling.BILINEAR)

        save_kwargs = {}
        if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 95

        resized.save(dst_path, **save_kwargs)
    return "ok"


def main() -> int:
    args = parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    scale = args.scale

    if not src.is_dir():
        raise SystemExit(f"[ERROR] source dir not found: {src}")
    if not (0 < scale <= 1):
        raise SystemExit(f"[ERROR] scale must be in (0, 1], got {scale}")

    dst.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise SystemExit(f"[ERROR] no images found in: {src}")

    print(f"[INFO] source={src}")
    print(f"[INFO] output={dst}")
    print(f"[INFO] scale={scale}")
    print(f"[INFO] workers={args.workers}")
    print(f"[INFO] overwrite={args.overwrite}")
    print(f"[INFO] total={len(files)}")

    done = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for src_path in files:
            dst_path = dst / src_path.name
            futs.append(ex.submit(resize_one, src_path, dst_path, scale, args.overwrite))

        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            if result == "skip":
                skipped += 1
            else:
                done += 1
            if i % 100 == 0 or i == len(futs):
                print(f"[INFO] progress {i}/{len(futs)} (written={done}, skipped={skipped})")

    print("[INFO] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
