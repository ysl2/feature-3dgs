import unittest
import numpy as np
import torch

from scene.cameras import Camera


class CameraMemoryBehaviorTest(unittest.TestCase):
    def test_original_image_stays_on_cpu_after_camera_init(self):
        image = torch.rand(3, 8, 8)
        cam = Camera(
            colmap_id=0,
            R=np.eye(3, dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            FoVx=1.0,
            FoVy=1.0,
            image=image,
            gt_alpha_mask=None,
            image_name="dummy",
            uid=0,
            semantic_feature=None,
        )

        self.assertEqual(cam.original_image.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
