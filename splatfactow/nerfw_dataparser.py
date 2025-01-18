# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phototourism dataset parser. Datasets and documentation here: http://phototour.cs.washington.edu/datasets/"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_binary,
    read_images_binary,
)
from nerfstudio.utils.rich_utils import CONSOLE

transient_objects = ['person', 'car', 'bicycle', 'minibike', 'tree', "desk",
                     "blanket", "bed ", "tray", "computer", "swimming pool",
                     "plate", "basket", "glass", "food", "land",
                     ]
label_id_mapping_ade20k = {'airplane': 90,
                           'animal': 126,
                           'apparel': 92,
                           'arcade machine': 78,
                           'armchair': 30,
                           'ashcan': 138,
                           'awning': 86,
                           'bag': 115,
                           'ball': 119,
                           'bannister': 95,
                           'bar': 77,
                           'barrel': 111,
                           'base': 40,
                           'basket': 112,
                           'bathtub': 37,
                           'bed ': 7,
                           'bench': 69,
                           'bicycle': 127,
                           'blanket': 131,
                           'blind': 63,
                           'boat': 76,
                           'book': 67,
                           'bookcase': 62,
                           'booth': 88,
                           'bottle': 98,
                           'box': 41,
                           'bridge': 61,
                           'buffet': 99,
                           'building': 1,
                           'bulletin board': 144,
                           'bus': 80,
                           'cabinet': 10,
                           'canopy': 106,
                           'car': 20,
                           'case': 55,
                           'ceiling': 5,
                           'chair': 19,
                           'chandelier': 85,
                           'chest of drawers': 44,
                           'clock': 148,
                           'coffee table': 64,
                           'column': 42,
                           'computer': 74,
                           'conveyer belt': 105,
                           'counter': 45,
                           'countertop': 70,
                           'cradle': 117,
                           'crt screen': 141,
                           'curtain': 18,
                           'cushion': 39,
                           'desk': 33,
                           'dirt track': 91,
                           'dishwasher': 129,
                           'door': 14,
                           'earth': 13,
                           'escalator': 96,
                           'fan': 139,
                           'fence': 32,
                           'field': 29,
                           'fireplace': 49,
                           'flag': 149,
                           'floor': 3,
                           'flower': 66,
                           'food': 120,
                           'fountain': 104,
                           'glass': 147,
                           'grandstand': 51,
                           'grass': 9,
                           'hill': 68,
                           'hood': 133,
                           'house': 25,
                           'hovel': 79,
                           'kitchen island': 73,
                           'lake': 128,
                           'lamp': 36,
                           'land': 94,
                           'light': 82,
                           'microwave': 124,
                           'minibike': 116,
                           'mirror': 27,
                           'monitor': 143,
                           'mountain': 16,
                           'ottoman': 97,
                           'oven': 118,
                           'painting': 22,
                           'palm': 72,
                           'path': 52,
                           'person': 12,
                           'pier': 140,
                           'pillow': 57,
                           'plant': 17,
                           'plate': 142,
                           'plaything': 108,
                           'pole': 93,
                           'pool table': 56,
                           'poster': 100,
                           'pot': 125,
                           'radiator': 146,
                           'railing': 38,
                           'refrigerator': 50,
                           'river': 60,
                           'road': 6,
                           'rock': 34,
                           'rug': 28,
                           'runway': 54,
                           'sand': 46,
                           'sconce': 134,
                           'screen': 130,
                           'screen door': 58,
                           'sculpture': 132,
                           'sea': 26,
                           'seat': 31,
                           'shelf': 24,
                           'ship': 103,
                           'shower': 145,
                           'sidewalk': 11,
                           'signboard': 43,
                           'sink': 47,
                           'sky': 2,
                           'skyscraper': 48,
                           'sofa': 23,
                           'stage': 101,
                           'stairs': 53,
                           'stairway': 59,
                           'step': 121,
                           'stool': 110,
                           'stove': 71,
                           'streetlight': 87,
                           'swimming pool': 109,
                           'swivel chair': 75,
                           'table': 15,
                           'tank': 122,
                           'television receiver': 89,
                           'tent': 114,
                           'toilet': 65,
                           'towel': 81,
                           'tower': 84,
                           'trade name': 123,
                           'traffic light': 136,
                           'tray': 137,
                           'tree': 4,
                           'truck': 83,
                           'van': 102,
                           'vase': 135,
                           'wall': 0,
                           'wardrobe': 35,
                           'washer': 107,
                           'water': 21,
                           'waterfall': 113,
                           'windowpane': 8}
id_label_mapping_ade20k = {v: k for k, v in label_id_mapping_ade20k.items()}

@dataclass
class NerfWDataParserConfig(DataParserConfig):
    """Phototourism dataset parser config"""

    _target: Type = field(default_factory=lambda: NerfW)
    """target class to instantiate"""
    data: Path = Path("data/brandenburg-gate")
    """Directory specifying location of data."""
    data_name: str = "brandenburg"
    """Name of the dataset."""
    scale_factor: float = 3.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    colmap_path: Path = Path("dense/sparse")
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction. This is helpful for Gaussian splatting and
    generally unused otherwise, but it's typically harmless so we default to True."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    max_2D_matches_per_3D_point: int = 0
    """Maximum number of 2D matches per 3D point. If set to -1, all 2D matches are loaded. If set to 0, no 2D matches are loaded."""
    include_semantics: bool = False
    """whether or not to include loading of semantics data"""

@dataclass
class NerfW(DataParser):
    """Phototourism dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: NerfWDataParserConfig

    def __init__(self, config: NerfWDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.i_eval = None

    def _generate_dataparser_outputs(self, split="train"):
        image_filenames = []
        poses = []
        colmap_path = self.data / self.config.colmap_path
        with CONSOLE.status(
            f"[bold green]Reading phototourism images and poses for {split} split..."
        ) as _:
            cams = read_cameras_binary(self.data / "dense/sparse/cameras.bin")
            imgs = read_images_binary(self.data / "dense/sparse/images.bin")

        poses = []
        fxs = []
        fys = []
        cxs = []
        cys = []
        image_filenames = []

        flip = torch.eye(3)
        flip[0, 0] = -1.0
        flip = flip.double()

        # Load the TSV file to get the train/eval split
        split_file = self.data / f"{self.config.data_name}.tsv"
        split_data = pd.read_csv(split_file, sep="\t")
        # kick lines that is NA
        split_data = split_data.dropna()

        for img_id, img in imgs.items():
            if img.name not in split_data[:]["filename"].values:
                continue
            cam = cams[img.camera_id]
            assert (
                cam.model == "PINHOLE"
            ), "Only pinhole (perspective) camera model is supported at the moment"

            pose = torch.cat(
                [torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))],
                dim=1,
            )
            pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
            poses.append(torch.linalg.inv(pose))
            fxs.append(torch.tensor(cam.params[0]))
            fys.append(torch.tensor(cam.params[1]))
            cxs.append(torch.tensor(cam.params[2]))
            cys.append(torch.tensor(cam.params[3]))

            image_filenames.append(self.data / "dense/images" / img.name)

        poses = torch.stack(poses).float()
        poses[..., 1:3] *= -1
        fxs = torch.stack(fxs).float()
        fys = torch.stack(fys).float()
        cxs = torch.stack(cxs).float()
        cys = torch.stack(cys).float()

        all_indices = torch.arange(len(image_filenames))
        # Create a mapping from image filenames to indices
        filename_to_index = {name: idx for idx, name in enumerate(image_filenames)}
        # Get the indices of the eval set in all indices
        eval_indices = [
            filename_to_index[self.data / "dense/images" / name]
            for name in split_data[split_data["split"] == "test"]["filename"].values
        ]

        eval_indices = torch.tensor(
            eval_indices,
            dtype=torch.long,
        )

        self.i_eval = eval_indices.tolist()
        # Print eval indices and corresponding filenames
        print(f"eval_indices: {eval_indices}")
        eval_filenames = [image_filenames[i] for i in eval_indices]
        print(f"eval_filenames: {eval_filenames}")

        if split == "train":
            indices = all_indices
        elif split == "val" or split == "test":
            indices = eval_indices
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            camera_type=CameraType.PERSPECTIVE,
        )
        cameras = cameras[indices]
        image_filenames = [image_filenames[i] for i in indices]
        metadata = {}
        if split == "train":
            metadata.update(
                self._load_3D_points(colmap_path, transform_matrix, scale_factor)
            )
        assert len(cameras) == len(image_filenames)
        # --- semantics ---
        if self.config.include_semantics:
            semantics_filenames = [
                (self.data / "semantic_maps" / img_path.name).with_suffix(".npz")
                for img_path in image_filenames
            ]
            classes = [id_label_mapping_ade20k[i] for i in range(len(id_label_mapping_ade20k))]
            semantics = Semantics(filenames=semantics_filenames, classes=classes, colors=None,
                                  mask_classes=transient_objects)
            metadata["semantics"] = semantics

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )

        return dataparser_outputs

    def _load_3D_points(
        self, colmap_path: Path, transform_matrix: torch.Tensor, scale_factor: float
    ):
        if (colmap_path / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(
                colmap_path / "points3D.bin"
            )
        elif (colmap_path / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(
                colmap_path / "points3D.txt"
            )
        else:
            raise ValueError(
                f"Could not find points3D.txt or points3D.bin in {colmap_path}"
            )
        points3D = torch.from_numpy(
            np.array([p.xyz for p in colmap_points.values()], dtype=np.float32)
        )
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor

        # Load point colours
        points3D_rgb = torch.from_numpy(
            np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8)
        )
        points3D_num_points = torch.tensor(
            [len(p.image_ids) for p in colmap_points.values()], dtype=torch.int64
        )
        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
            "points3D_error": torch.from_numpy(
                np.array([p.error for p in colmap_points.values()], dtype=np.float32)
            ),
            "points3D_num_points2D": points3D_num_points,
        }
        if self.config.max_2D_matches_per_3D_point != 0:
            if (colmap_path / "images.txt").exists():
                im_id_to_image = colmap_utils.read_images_text(
                    colmap_path / "images.txt"
                )
            elif (colmap_path / "images.bin").exists():
                im_id_to_image = colmap_utils.read_images_binary(
                    colmap_path / "images.bin"
                )
            else:
                raise ValueError(
                    f"Could not find images.txt or images.bin in {colmap_path}"
                )
            downscale_factor = self._downscale_factor
            max_num_points = int(torch.max(points3D_num_points).item())
            if self.config.max_2D_matches_per_3D_point > 0:
                max_num_points = min(
                    max_num_points, self.config.max_2D_matches_per_3D_point
                )
            points3D_image_ids = []
            points3D_image_xy = []
            for p in colmap_points.values():
                nids = np.array(p.image_ids, dtype=np.int64)
                nxy_ids = np.array(p.point2D_idxs, dtype=np.int32)
                if self.config.max_2D_matches_per_3D_point != -1:
                    # Randomly sample 2D matches
                    idxs = np.argsort(p.error)[
                        : self.config.max_2D_matches_per_3D_point
                    ]
                    nids = nids[idxs]
                    nxy_ids = nxy_ids[idxs]
                nxy = [
                    im_id_to_image[im_id].xys[pt_idx]
                    for im_id, pt_idx in zip(nids, nxy_ids)
                ]
                nxy = torch.from_numpy(np.stack(nxy).astype(np.float32))
                nids = torch.from_numpy(nids)
                assert len(nids.shape) == 1
                assert len(nxy.shape) == 2
                points3D_image_ids.append(
                    torch.cat(
                        (
                            nids,
                            torch.full(
                                (max_num_points - len(nids),), -1, dtype=torch.int64
                            ),
                        )
                    )
                )
                points3D_image_xy.append(
                    torch.cat(
                        (
                            nxy,
                            torch.full(
                                (max_num_points - len(nxy), nxy.shape[-1]),
                                0,
                                dtype=torch.float32,
                            ),
                        )
                    )
                    / downscale_factor
                )
            out["points3D_image_ids"] = torch.stack(points3D_image_ids, dim=0)
            out["points3D_points2D_xy"] = torch.stack(points3D_image_xy, dim=0)
        return out

    def check_in_eval(self, idx):
        return idx in self.i_eval

    def find_eval_idx(self, idx):
        return self.i_eval[idx]


splatfactow_dataparser=DataParserSpecification(config=NerfWDataParserConfig())
