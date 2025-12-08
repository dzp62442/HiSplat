import json
import os.path as osp
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import repeat
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler
from .utils_omniscene import load_conditions, load_info


@dataclass
class DatasetOmniSceneCfg(DatasetCfgCommon):
    name: Literal["omniscene"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    test_chunk_interval: int = 1
    test_times_per_scene: int = 1
    train_times_per_scene: int = 1
    highres: bool = False


class DatasetOmniScene(Dataset):
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        cfg: DatasetOmniSceneCfg,
        stage: Stage,
        view_sampler: ViewSampler | None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.resolution = tuple(cfg.image_shape)
        self.data_root = str(cfg.roots[0])
        self.data_version = "interp_12Hz_trainval"
        self.dataset_prefix = "/datasets/nuScenes"
        self.near = cfg.near if cfg.near > 0 else 0.5
        self.far = cfg.far if cfg.far > 0 else 100.0

        self.bin_tokens = self._load_bin_tokens(stage)

    def _load_bin_tokens(self, stage: Stage) -> list[str]:
        version_root = osp.join(self.data_root, self.data_version)
        if stage == "train":
            info = osp.join(version_root, "bins_train_3.2m.json")
            return json.load(open(info))["bins"]
        if stage == "val":
            info = osp.join(version_root, "bins_val_3.2m.json")
            bins = json.load(open(info))["bins"]
            return bins[:30000:3000][:10]
        if stage == "test":
            info = osp.join(version_root, "bins_val_3.2m.json")
            bins = json.load(open(info))["bins"]
            return bins[0::14][:2048]
        raise ValueError(f"Unsupported stage: {stage}")

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def __getitem__(self, index: int):
        bin_token = self.bin_tokens[index]
        bin_info_path = osp.join(
            self.data_root,
            self.data_version,
            "bin_infos_3.2m",
            f"{bin_token}.pkl",
        )
        with open(bin_info_path, "rb") as fh:
            bin_info = pkl.load(fh)

        sensor_info_center = {
            sensor: bin_info["sensor_info"][sensor][0]
            for sensor in self.camera_types + ["LIDAR_TOP"]
        }

        input_paths, input_c2w = [], []
        for cam in self.camera_types:
            info = sensor_info_center[cam]
            img_path, c2w, _ = load_info(info)
            img_path = img_path.replace(self.dataset_prefix, self.data_root)
            input_paths.append(img_path)
            input_c2w.append(c2w)
        input_c2w = torch.as_tensor(input_c2w, dtype=torch.float32)
        input_imgs, input_masks, input_intr = load_conditions(input_paths, self.resolution, is_input=True)

        output_paths, output_c2w = [], []
        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        if frame_num < 3:
            raise ValueError(f"Bin {bin_token} does not contain enough frames")

        rend_indices = [[1, 2]] * len(self.camera_types)
        for cam_id, cam in enumerate(self.camera_types):
            for offset in rend_indices[cam_id]:
                info = bin_info["sensor_info"][cam][offset]
                img_path, c2w, _ = load_info(info)
                img_path = img_path.replace(self.dataset_prefix, self.data_root)
                output_paths.append(img_path)
                output_c2w.append(c2w)
        output_c2w = torch.as_tensor(output_c2w, dtype=torch.float32)
        output_imgs, output_masks, output_intr = load_conditions(output_paths, self.resolution, is_input=False)

        output_imgs = torch.cat([output_imgs, input_imgs], dim=0)
        output_masks = torch.cat([output_masks, input_masks], dim=0)
        output_c2w = torch.cat([output_c2w, input_c2w], dim=0)
        output_intr = torch.cat([output_intr, input_intr], dim=0)

        context = {
            "extrinsics": input_c2w,
            "intrinsics": input_intr,
            "image": input_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=len(input_c2w)),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=len(input_c2w)),
            "index": torch.arange(len(input_c2w), dtype=torch.long),
        }
        target = {
            "extrinsics": output_c2w,
            "intrinsics": output_intr,
            "image": output_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=len(output_c2w)),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=len(output_c2w)),
            "index": torch.arange(len(output_c2w), dtype=torch.long),
            "masks": output_masks,
        }
        return {
            "context": context,
            "target": target,
            "scene": bin_token,
        }
