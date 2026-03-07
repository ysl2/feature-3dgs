import os
import unittest
from unittest import mock

from encoders.lseg_encoder.modules.models import lseg_vit


class ClipCacheConfigTest(unittest.TestCase):
    def test_default_download_root_is_user_cache(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLIP_DOWNLOAD_ROOT", None)
            root = lseg_vit._clip_download_root()
            self.assertEqual(root, os.path.expanduser("~/.cache/clip"))

    def test_env_override_download_root(self):
        with mock.patch.dict(os.environ, {"CLIP_DOWNLOAD_ROOT": "/tmp/custom-clip-cache"}, clear=False):
            root = lseg_vit._clip_download_root()
            self.assertEqual(root, "/tmp/custom-clip-cache")


if __name__ == "__main__":
    unittest.main()
