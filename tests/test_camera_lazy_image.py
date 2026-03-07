import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from scene.cameras import Camera
from utils.camera_utils import loadCam


class CameraLazyImageLoadingTest(unittest.TestCase):
    def _make_png(self, path: Path, width: int = 16, height: int = 8):
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[..., 0] = 255
        Image.fromarray(arr, mode="RGB").save(path)

    def test_camera_can_lazy_load_original_image_from_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "frame.png"
            self._make_png(image_path)

            cam = Camera(
                colmap_id=0,
                R=np.eye(3, dtype=np.float32),
                T=np.zeros(3, dtype=np.float32),
                FoVx=1.0,
                FoVy=1.0,
                image=None,
                gt_alpha_mask=None,
                image_name="frame",
                uid=0,
                semantic_feature=None,
                image_path=str(image_path),
                image_resolution=(16, 8),
            )

            original_image = cam.original_image
            self.assertEqual(original_image.device.type, "cpu")
            self.assertEqual(tuple(original_image.shape), (3, 8, 16))

    def test_loadCam_does_not_preload_colmap_image_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "frame.png"
            self._make_png(image_path, width=20, height=10)

            cam_info = SimpleNamespace(
                uid=1,
                R=np.eye(3, dtype=np.float32),
                T=np.zeros(3, dtype=np.float32),
                FovY=1.0,
                FovX=1.0,
                image=None,
                image_path=str(image_path),
                image_name="frame",
                width=20,
                height=10,
                semantic_feature=None,
                semantic_feature_path=None,
                semantic_feature_shape=(1, 10, 20),
            )
            args = SimpleNamespace(resolution=1, data_device="cuda")

            cam = loadCam(args, id=0, cam_info=cam_info, resolution_scale=1.0)
            self.assertIsNone(cam._original_image)
            self.assertEqual(tuple(cam.original_image.shape), (3, 10, 20))


if __name__ == "__main__":
    unittest.main()
