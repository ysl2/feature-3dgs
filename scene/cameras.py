#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, semantic_feature, semantic_feature_path=None,
                 image_path=None, image_resolution=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self._semantic_feature = semantic_feature
        self.semantic_feature_path = semantic_feature_path

        self.image_path = image_path
        self._original_image = None
        self._gt_alpha_mask = gt_alpha_mask

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # Keep GT image on CPU and avoid preloading all camera tensors into RAM.
        if image is not None:
            self._original_image = image.clamp(0.0, 1.0)
            self.image_width = self._original_image.shape[2]
            self.image_height = self._original_image.shape[1]

            if gt_alpha_mask is not None:
                self._original_image *= gt_alpha_mask
            else:
                self._original_image *= torch.ones((1, self.image_height, self.image_width))
        else:
            if image_path is None or image_resolution is None:
                raise RuntimeError(f"image_path/image_resolution missing for camera {self.image_name}")
            self.image_width = int(image_resolution[0])
            self.image_height = int(image_resolution[1])

        self.width = self.image_width
        self.height = self.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar,
                                                     fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def original_image(self):
        # Preloaded path (e.g., Blender) keeps previous behavior.
        if self._original_image is not None:
            return self._original_image

        # Colmap path: lazy-load from disk on demand to avoid scene-init RAM spikes.
        if self.image_path is None:
            raise RuntimeError(f"image_path missing for camera {self.image_name}")

        with Image.open(self.image_path) as image_pil:
            resized_image_rgb = PILtoTorch(image_pil, (self.image_width, self.image_height))

        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None

        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
        elif self._gt_alpha_mask is not None:
            loaded_mask = self._gt_alpha_mask

        if loaded_mask is not None:
            gt_image *= loaded_mask

        return gt_image

    @property
    def semantic_feature(self):
        # Lazy-load semantic feature to avoid preloading all per-image feature maps into RAM.
        if self._semantic_feature is not None:
            return self._semantic_feature
        if self.semantic_feature_path is None:
            raise RuntimeError(f"semantic_feature_path missing for camera {self.image_name}")
        return torch.load(self.semantic_feature_path, map_location="cpu")


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.projection_matrix = torch.bmm(self.world_view_transform.unsqueeze(0).inverse(),
                                           self.full_proj_transform.unsqueeze(0)).squeeze(0)
