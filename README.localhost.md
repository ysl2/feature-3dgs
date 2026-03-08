# Localhost Viewer Notes (Spark-based preview)

## Goal
Use Spark as a **minimal-change** local viewer for Feature-3DGS outputs, with separate RGB and semantic preview pages.

## Code tracked in this repo
### Export script
- `scripts/export_spark_preview.py`
  - Reads Feature-3DGS `point_cloud.ply` (binary_little_endian)
  - Detects `semantic_*` channels
  - Exports Spark-friendly preview PLYs
  - Supports:
    - PCA semantic color preview (`--out-pca`)
    - RGB preview from `f_dc_0/1/2` (`--out-rgb`)
    - Optional query-vector heatmap (`--query-vector`, `--out-query`)

### Tests
- `tests/test_export_spark_preview.py`
  - unittest coverage for `sigmoid`, PCA color export shape/range, and RGB-from-DC conversion

## Runtime assets (external SSD, not in git)
Base directory:
- `/media/songliyu/T7_Shield/Documents/feature-3dgs/output/DJI_0544--lseg-7000/point_cloud/iteration_7000`

Generated preview PLYs:
- `rgb_preview_500k.ply`
- `semantic_pca_preview_500k.ply`
- `rgb_preview_full.ply` (via direct load of `point_cloud.ply` in page)
- `semantic_pca_preview_full.ply`

Preview pages:
- `rgb_preview.html` (500k RGB)
- `rgb_preview_full.html` (full RGB)
- `semantic_preview.html` (500k semantic PCA)
- `semantic_preview_full.html` (full semantic PCA)

## Local run
Serve preview directory:
```bash
python3 -m http.server 8787 --directory /media/songliyu/T7_Shield/Documents/feature-3dgs/output/DJI_0544--lseg-7000/point_cloud/iteration_7000
```

Then open:
- `http://127.0.0.1:8787/rgb_preview.html`
- `http://127.0.0.1:8787/rgb_preview_full.html`
- `http://127.0.0.1:8787/semantic_preview.html`
- `http://127.0.0.1:8787/semantic_preview_full.html`

## Current interaction tweaks applied in page
- White background
- Vertical orientation fixed (`rotation.x = Math.PI`)
- Horizontal drag direction corrected (yaw sign flipped)

## Related repositories
- Feature-3DGS code: `/home/songliyu/Documents/feature-3dgs`
- Spark upstream clone for reference: `/home/songliyu/Documents/spark`
- Splat reference clone: `/home/songliyu/Documents/splat`

## Relevant commits in `feature-3dgs`
- `e2cb07d` feat: export spark-ready semantic PCA preview ply from feature-3dgs point cloud
- `df8880d` feat: add rgb preview export from feature-3dgs f_dc colors
