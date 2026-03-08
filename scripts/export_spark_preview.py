#!/usr/bin/env python3
"""Export lightweight preview PLY files for Spark viewer from feature-3dgs point clouds.

This script intentionally avoids touching Spark internals. It reads the feature-3dgs
point cloud PLY, keeps geometry/splat params, and rewrites only display colors.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


SH_C0 = 0.28209479177387814

PLY_TO_NUMPY = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
}


@dataclass
class PlyMeta:
    count: int
    dtype: np.dtype
    data_offset: int
    semantic_fields: List[str]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pca_rgb(semantic: np.ndarray, fit_samples: int = 50000) -> np.ndarray:
    if semantic.ndim != 2:
        raise ValueError(f"semantic must be 2D, got shape={semantic.shape}")
    n = semantic.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    sample_n = min(n, fit_samples)
    if sample_n < n:
        idx = np.random.default_rng(0).choice(n, size=sample_n, replace=False)
        fit = semantic[idx]
    else:
        fit = semantic

    mean = fit.mean(axis=0, keepdims=True)
    centered_fit = fit - mean
    _, _, vt = np.linalg.svd(centered_fit, full_matrices=False)
    comps = vt[:3].T  # [D,3]

    proj = (semantic - mean) @ comps

    lo = np.percentile(proj, 1, axis=0, keepdims=True)
    hi = np.percentile(proj, 99, axis=0, keepdims=True)
    denom = np.maximum(hi - lo, 1e-6)
    norm = np.clip((proj - lo) / denom, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def rgb_from_dc(vertices: np.ndarray) -> np.ndarray:
    names = set(vertices.dtype.names or [])
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        r = np.clip(vertices["f_dc_0"].astype(np.float32) * SH_C0 + 0.5, 0.0, 1.0)
        g = np.clip(vertices["f_dc_1"].astype(np.float32) * SH_C0 + 0.5, 0.0, 1.0)
        b = np.clip(vertices["f_dc_2"].astype(np.float32) * SH_C0 + 0.5, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=1)
        return (rgb * 255.0).astype(np.uint8)

    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack(
            [
                vertices["red"].astype(np.float32),
                vertices["green"].astype(np.float32),
                vertices["blue"].astype(np.float32),
            ],
            axis=1,
        )
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    return np.full((vertices.shape[0], 3), 255, dtype=np.uint8)


def dot_rgb(semantic: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    if query_vec.ndim != 1:
        raise ValueError("query_vec must be 1D")
    if semantic.shape[1] != query_vec.shape[0]:
        raise ValueError(
            f"dim mismatch: semantic D={semantic.shape[1]} vs query D={query_vec.shape[0]}"
        )
    q = query_vec.astype(np.float32)
    q = q / max(np.linalg.norm(q), 1e-8)

    s = semantic.astype(np.float32)
    s = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-8)
    score = (s @ q).astype(np.float32)

    lo = np.percentile(score, 1)
    hi = np.percentile(score, 99)
    x = np.clip((score - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    rgb = np.zeros((score.shape[0], 3), dtype=np.uint8)
    rgb[:, 0] = (255 * x).astype(np.uint8)  # red high
    rgb[:, 2] = (255 * (1.0 - x)).astype(np.uint8)  # blue low
    rgb[:, 1] = (255 * (1.0 - np.abs(x - 0.5) * 2.0) * 0.3).astype(np.uint8)
    return rgb


def parse_ply_meta(path: str) -> PlyMeta:
    with open(path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PLY header")
            s = line.decode("ascii", errors="strict").strip()
            header_lines.append(s)
            if s == "end_header":
                break
        data_offset = f.tell()

    if not header_lines or header_lines[0] != "ply":
        raise RuntimeError("Not a PLY file")
    fmt = next((l for l in header_lines if l.startswith("format ")), None)
    if fmt != "format binary_little_endian 1.0":
        raise RuntimeError(f"Unsupported format: {fmt}")

    in_vertex = False
    count = None
    fields: List[Tuple[str, np.dtype]] = []
    semantic_fields: List[str] = []

    for line in header_lines:
        if line.startswith("element "):
            parts = line.split()
            name = parts[1]
            if name == "vertex":
                in_vertex = True
                count = int(parts[2])
            else:
                in_vertex = False
            continue
        if in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) != 3:
                raise RuntimeError(f"Unsupported property line: {line}")
            typ, name = parts[1], parts[2]
            if typ not in PLY_TO_NUMPY:
                raise RuntimeError(f"Unsupported PLY property type: {typ}")
            fields.append((name, PLY_TO_NUMPY[typ]))
            if name.startswith("semantic_"):
                semantic_fields.append(name)

    if count is None:
        raise RuntimeError("PLY missing vertex element")
    if not semantic_fields:
        raise RuntimeError("No semantic_* fields found in vertex data")

    return PlyMeta(
        count=count,
        dtype=np.dtype(fields),
        data_offset=data_offset,
        semantic_fields=semantic_fields,
    )


def load_selected_vertices(path: str, meta: PlyMeta, max_points: int) -> np.ndarray:
    mm = np.memmap(
        path,
        mode="r",
        dtype=meta.dtype,
        offset=meta.data_offset,
        shape=(meta.count,),
    )
    if max_points <= 0 or max_points >= meta.count:
        return np.array(mm)
    idx = np.random.default_rng(0).choice(meta.count, size=max_points, replace=False)
    idx.sort()
    return np.array(mm[idx])


def write_preview_ply(path: str, vertices: np.ndarray, rgb: np.ndarray) -> None:
    out_dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("scale_0", np.float32),
            ("scale_1", np.float32),
            ("scale_2", np.float32),
            ("rot_0", np.float32),
            ("rot_1", np.float32),
            ("rot_2", np.float32),
            ("rot_3", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
            ("alpha", np.uint8),
        ]
    )
    out = np.empty(vertices.shape[0], dtype=out_dtype)

    for k in ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]:
        out[k] = vertices[k].astype(np.float32)

    out["red"] = rgb[:, 0]
    out["green"] = rgb[:, 1]
    out["blue"] = rgb[:, 2]

    if "opacity" in vertices.dtype.names:
        alpha = np.clip(sigmoid(vertices["opacity"].astype(np.float32)) * 255.0, 0, 255).astype(np.uint8)
    else:
        alpha = np.full(vertices.shape[0], 255, dtype=np.uint8)
    out["alpha"] = alpha

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {out.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property uchar alpha",
            "end_header",
            "",
        ]
    ).encode("ascii")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        out.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Spark preview PLY from feature-3dgs point cloud")
    parser.add_argument("--input", required=True, help="Input point_cloud.ply path")
    parser.add_argument("--out-pca", required=True, help="Output PLY path for PCA color preview")
    parser.add_argument("--max-points", type=int, default=500000, help="Randomly sample at most N points for interactive preview")
    parser.add_argument("--query-vector", default=None, help="Optional .npy query vector (D,) to export dot-product heatmap")
    parser.add_argument("--out-query", default=None, help="Output PLY path for query heatmap (required with --query-vector)")
    parser.add_argument("--out-rgb", default=None, help="Optional output PLY path for original RGB preview (from f_dc_*)")
    args = parser.parse_args()

    meta = parse_ply_meta(args.input)
    vertices = load_selected_vertices(args.input, meta, args.max_points)
    semantic = np.stack([vertices[name].astype(np.float32) for name in meta.semantic_fields], axis=1)

    rgb_pca = pca_rgb(semantic)
    write_preview_ply(args.out_pca, vertices, rgb_pca)
    print(f"[OK] PCA preview written: {args.out_pca} ({vertices.shape[0]} points)")

    if args.out_rgb:
        rgb_native = rgb_from_dc(vertices)
        write_preview_ply(args.out_rgb, vertices, rgb_native)
        print(f"[OK] RGB preview written: {args.out_rgb}")

    if args.query_vector is not None:
        if not args.out_query:
            raise ValueError("--out-query is required when --query-vector is provided")
        q = np.load(args.query_vector)
        rgb_q = dot_rgb(semantic, q)
        write_preview_ply(args.out_query, vertices, rgb_q)
        print(f"[OK] Query preview written: {args.out_query}")


if __name__ == "__main__":
    main()
