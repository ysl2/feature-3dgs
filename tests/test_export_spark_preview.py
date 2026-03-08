import unittest
import numpy as np

from scripts.export_spark_preview import pca_rgb, sigmoid


class TestExportSparkPreview(unittest.TestCase):
    def test_sigmoid_basic(self):
        x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
        y = sigmoid(x)
        self.assertLess(y[0], 1e-4)
        self.assertAlmostEqual(float(y[1]), 0.5, places=6)
        self.assertGreater(y[2], 0.9999)

    def test_pca_rgb_shape_and_range(self):
        rng = np.random.default_rng(0)
        semantic = rng.normal(size=(200, 16)).astype(np.float32)
        rgb = pca_rgb(semantic, fit_samples=100)
        self.assertEqual(rgb.shape, (200, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreaterEqual(int(rgb.min()), 0)
        self.assertLessEqual(int(rgb.max()), 255)


if __name__ == "__main__":
    unittest.main()
