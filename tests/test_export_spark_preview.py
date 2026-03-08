import unittest
import numpy as np

from scripts.export_spark_preview import pca_rgb, sigmoid, rgb_from_dc


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

    def test_rgb_from_dc_shape_and_range(self):
        dt = np.dtype([
            ("f_dc_0", np.float32),
            ("f_dc_1", np.float32),
            ("f_dc_2", np.float32),
        ])
        verts = np.zeros(5, dtype=dt)
        verts["f_dc_0"] = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
        verts["f_dc_1"] = 0.0
        verts["f_dc_2"] = 1.0
        rgb = rgb_from_dc(verts)
        self.assertEqual(rgb.shape, (5, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreaterEqual(int(rgb.min()), 0)
        self.assertLessEqual(int(rgb.max()), 255)


if __name__ == "__main__":
    unittest.main()
