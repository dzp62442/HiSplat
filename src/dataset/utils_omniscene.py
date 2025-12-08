import json
from typing import Sequence

import numpy as np
from PIL import Image
import torch


def _ensure_hwc3(array: np.ndarray) -> np.ndarray:
    """Ensure the input array has 3 channels (RGB) in HWC format."""
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    if array.shape[2] == 3:
        return array
    if array.shape[2] == 1:
        return np.repeat(array, 3, axis=2)
    if array.shape[2] == 4:
        color = array[:, :, :3].astype(np.float32)
        alpha = array[:, :, 3:4].astype(np.float32) / 255.0
        blended = color * alpha + 255.0 * (1.0 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)
    raise ValueError("Unsupported channel size for image conversion")


def load_info(info: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """Load image path and camera transforms from sensor info."""
    img_path = info["data_path"]
    c2w = info["sensor2lidar_transform"]

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t

    return img_path, c2w, w2c


def _maybe_resize_image(img: Image.Image, target_reso: Sequence[int], intrinsics: np.ndarray):
    if img.height == target_reso[0] and img.width == target_reso[1]:
        return np.array(img), intrinsics, False

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    scale_h, scale_w = target_reso[0] / img.height, target_reso[1] / img.width
    fx *= scale_w
    fy *= scale_h
    cx *= scale_w
    cy *= scale_h
    scaled_intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    resized = img.resize((target_reso[1], target_reso[0]))
    return np.array(resized), scaled_intrinsics, True


def load_conditions(img_paths, reso, is_input: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, masks, intrinsics = [], [], []
    for img_path in img_paths:
        param_path = img_path.replace("samples", "samples_param_small")
        param_path = param_path.replace("sweeps", "sweeps_param_small")
        param_path = param_path.replace(".jpg", ".json")
        params = json.load(open(param_path))
        ck = np.array(params["camera_intrinsic"], dtype=np.float32)

        disk_path = img_path.replace("samples", "samples_small").replace("sweeps", "sweeps_small")
        img = Image.open(disk_path)
        img_np, ck_scaled, resized = _maybe_resize_image(img, reso, ck)

        ck_scaled[0, :] /= reso[1]
        ck_scaled[1, :] /= reso[0]
        images.append(_ensure_hwc3(img_np))
        intrinsics.append(ck_scaled.astype(np.float32))

        if is_input:
            mask = np.ones(tuple(reso), dtype=np.float32)
        else:
            mask_path = disk_path.replace("sweeps_small", "sweeps_mask_small").replace("samples_small", "samples_mask_small")
            mask_path = mask_path.replace(".jpg", ".png")
            mask_img = Image.open(mask_path).convert("L")
            if resized:
                mask_img = mask_img.resize((reso[1], reso[0]), Image.BILINEAR)
            mask = np.array(mask_img).astype(np.float32) / 255.0
        masks.append(mask)

    images_tensor = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    masks_tensor = torch.from_numpy(np.stack(masks, axis=0)).bool()
    intrinsics_tensor = torch.as_tensor(np.stack(intrinsics, axis=0), dtype=torch.float32)
    return images_tensor, masks_tensor, intrinsics_tensor
