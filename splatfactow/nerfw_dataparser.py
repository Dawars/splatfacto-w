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
import yaml
from rich.progress import track
from typing_extensions import Literal

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
    read_points3D_binary,
)
from nerfstudio.model_components.ray_samplers import save_points

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

    _target: Type = field(default_factory=lambda: Heritage)
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
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    max_2D_matches_per_3D_point: int = 0
    """Maximum number of 2D matches per 3D point. If set to -1, all 2D matches are loaded. If set to 0, no 2D matches are loaded."""
    include_semantics: bool = False
    """whether or not to include loading of semantics data"""
    setting: str = ""
    """Choose tsv file which contains subset of images, e.g: all, clean, facade"""


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
class Heritage(DataParser):
    """Phototourism dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: NerfWDataParserConfig

    def __init__(self, config: NerfWDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.train_split_percentage = config.train_split_percentage

    # pylint: disable=too-many-statements
    def _generate_dataparser_outputs(self, split="train"):

        setting_suffix = '' if self.config.setting == '' else f'_{self.config.setting}'
        config_path = self.data / f"config{setting_suffix}.yaml"
        print(f"Config file: {str(config_path)}")

        with open(config_path, "r") as yamlfile:
            scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

        sfm_to_gt = np.array(scene_config["sfm2gt"])
        gt_to_sfm = np.linalg.inv(sfm_to_gt)
        sfm_vert1 = gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][0]) + gt_to_sfm[:3, 3]
        sfm_vert2 = gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][1]) + gt_to_sfm[:3, 3]
        bbx_min = np.minimum(sfm_vert1, sfm_vert2)
        bbx_max = np.maximum(sfm_vert1, sfm_vert2)

        with CONSOLE.status(f"[bold green]Reading phototourism images and poses for {split} split...") as _:
            cams = read_cameras_binary(self.data / "dense/sparse/cameras.bin")
            imgs = read_images_binary(self.data / "dense/sparse/images.bin")
            pts3d = read_points3D_binary(self.data / "dense/sparse/points3D.bin")

        img_path_to_id = {}
        file_list = []
        image_list = list(self.data.glob(f"*{setting_suffix}.tsv"))
        if image_list:
            print(f"Found .tsv file for image list {image_list[0]}")
            self.files = pd.read_csv(image_list[0], sep="\t")
            self.files = self.files[~self.files['id'].isnull()]  # remove data without id
            self.files.reset_index(inplace=True, drop=True)
            file_list = list(self.files["filename"])

            for v in imgs.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {}  # {id: filename}
            for filename in list(self.files['filename']):
                if filename not in img_path_to_id:
                    continue
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]
            # for v in imgs.values():
            #     if v.name in file_list:
            #         img_path_to_id[v.name] = v.id
        else:
            raise f"Image list not found *{setting_suffix}.tsv"
            # for _id, cam in cams.items():
            #     img = imgs[_id]
            #     img_path_to_id[img.name] = img.id
            #     file_list.append(img.name)
        # file_list = sorted(file_list)

        # key point depth
        pts3d_array = torch.ones(max(pts3d.keys()) + 1, 4)
        error_array = torch.ones(max(pts3d.keys()) + 1, 1)
        for pts_id, pts in track(pts3d.items(), description="create 3D points", transient=True):
            pts3d_array[pts_id, :3] = torch.from_numpy(pts.xyz)
            error_array[pts_id, 0] = torch.from_numpy(pts.error)

        # determine mask extension
        mask_ext = ".npy" if list((self.data / "masks").glob("*.npy")) else ".png"

        poses = []
        fxs = []
        fys = []
        cxs = []
        cys = []
        heights = []
        widths = []
        image_filenames = []
        mask_filenames = []
        semantic_filenames = []
        sparse_pts = []

        for filename in file_list:
            if filename not in img_path_to_id.keys():
                print(f"image {filename} not found in sfm result!!")
                continue
            _id = img_path_to_id[filename]
            img = imgs[_id]
            cam = cams[img.camera_id]

            assert cam.model == "PINHOLE", "Only pinhole (perspective) camera model is supported at the moment"

            pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
            pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
            poses.append(torch.linalg.inv(pose))
            fxs.append(torch.tensor(cam.params[0]))
            fys.append(torch.tensor(cam.params[1]))
            cxs.append(torch.tensor(cam.params[2]))
            cys.append(torch.tensor(cam.params[3]))
            heights.append(torch.tensor(cam.height))
            widths.append(torch.tensor(cam.width))

            image_filenames.append(self.data / "dense/images" / img.name)
            mask_filenames.append(self.data / "masks" / img.name.replace(".jpg", mask_ext))
            semantic_filenames.append(self.data / "semantic_maps" / img.name.replace(".jpg", ".npz"))
            # if self.config.include_mono_prior:
            #     depth_filenames.append(self.data / "depth" / img.name.replace(".jpg", self.config.depth_extension))
            #     sensor_filenames.append(self.data / "dense" / "stereo" / "depth_maps" / img.name.replace(".jpg", ".jpg.geometric.bin"))
            #     normal_filenames.append(self.data / "normal" / img.name.replace(".jpg", ".npy"))

            # load sparse 3d points for each view
            # visualize pts3d for each image
            valid_3d_mask = img.point3D_ids != -1
            point3d_ids = img.point3D_ids[valid_3d_mask]
            img_p3d = pts3d_array[point3d_ids]
            img_err = error_array[point3d_ids]
            # img_p3d = img_p3d[img_err[:, 0] < torch.median(img_err)]

            # weight term as in NeuralRecon-W
            err_mean = img_err.mean()
            weight = 2 * np.exp(-((img_err / err_mean) ** 2))

            img_p3d[:, 3:] = weight

            sparse_pts.append(img_p3d)

        poses = torch.stack(poses).float()
        poses[..., 1:3] *= -1
        fxs = torch.stack(fxs).float()
        fys = torch.stack(fys).float()
        cxs = torch.stack(cxs).float()
        cys = torch.stack(cys).float()
        heights = torch.stack(heights)
        widths = torch.stack(widths)

        # filter image_filenames and poses based on train/eval split percentage

        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        i_train = [i for i, filename in enumerate(image_filenames)
                   if self.files.loc[i, 'split'] == 'train']
        i_eval = [i for i, filename in enumerate(image_filenames)
                  if self.files.loc[i, 'split'] == 'test']
        self.i_eval = i_eval

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        """
        poses = camera_utils.auto_orient_and_center_poses(
            poses, method=self.config.orientation_method, center_poses=self.config.center_poses
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(poses[:, :3, 3]))

        poses[:, :3, 3] *= scale_factor * self.config.scale_factor
        # shift back so that the object is aligned?
        poses[:, 1, 3] -= 1
        """

        # normalize with scene radius
        radius = scene_config["radius"]
        scale = 1.0 / (radius * 1.01)  # enlarge the radius a little bit
        origin = np.array(scene_config["origin"]).reshape(1, 3)
        origin = torch.from_numpy(origin)
        poses[:, :3, 3] -= origin
        poses[:, :3, 3] *= scale

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses, method=self.config.orientation_method, center_method=self.config.center_method
        )

        # scale pts accordingly
        for pts in sparse_pts:
            pts[:, :3] -= origin
            pts[:, :3] *= scale  # should be the same as pose preprocessing
            pts[:, :3] = pts[:, :3] @ transform_matrix[:3, :3].t() + transform_matrix[:3, 3:].t()

        # create occupancy grid from sparse points
        points_ori = []
        min_track_length = scene_config["min_track_length"]
        for _, p in pts3d.items():
            if p.point2D_idxs.shape[0] > min_track_length:
                points_ori.append(p.xyz)
        points_ori = np.array(points_ori)
        save_points("nori_10.ply", points_ori)

        # filter with bbox
        # normalize cropped area to [-1, -1]
        scene_origin = origin.numpy()

        points_normalized = (points_ori - scene_origin) / (bbx_max - bbx_min) * 2
        # filter out points out of [-1, 1]
        mask = np.prod((points_normalized > -1), axis=-1, dtype=bool) & np.prod(
            (points_normalized < 1), axis=-1, dtype=bool
        )
        points_ori = points_ori[mask]

        save_points("nori_10_filterbbox.ply", points_ori)

        points_ori = torch.from_numpy(points_ori).float()

        # scale pts accordingly
        points_ori -= origin
        points_ori[:, :3] *= scale  # should be the same as pose preprocessing
        points_ori[:, :3] = points_ori[:, :3] @ transform_matrix[:3, :3].t() + transform_matrix[:3, 3:].t()

        # expand and quantify

        offset = torch.linspace(-1, 1.0, 3)
        offset_cube = torch.meshgrid(offset, offset, offset)
        offset_cube = torch.stack(offset_cube, dim=-1).reshape(-1, 3)

        voxel_size = scene_config["voxel_size"] / (radius * 1.01)
        # level = int(np.floor(np.log2(2 * scale / voxel_size)))

        offset_cube *= voxel_size  # voxel size
        expand_points = points_ori[:, None, :] + offset_cube[None]
        expand_points = expand_points.reshape(-1, 3)
        save_points("expand_points.ply", expand_points.numpy())

        # filter
        # filter out points out of [-1, 1]
        mask = torch.prod((expand_points > -1.0), axis=-1, dtype=torch.bool) & torch.prod(
            (expand_points < 1.0), axis=-1, dtype=torch.bool
        )
        filtered_points = expand_points[mask]
        save_points("filtered_points.ply", filtered_points.numpy())

        grid_size = 32
        voxel_size = 2.0 / grid_size
        quantified_points = torch.floor((filtered_points + 1.0) * grid_size // 2)

        index = quantified_points[:, 0] * grid_size**2 + quantified_points[:, 1] * grid_size + quantified_points[:, 2]

        offset = torch.linspace(-1.0 + voxel_size / 2.0, 1.0 - voxel_size / 2.0, grid_size)
        x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
        offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        # xyz
        mask = torch.zeros(grid_size**3, dtype=torch.bool)
        mask[index.long()] = True

        points_valid = offset_cube[mask]
        save_points("quantified_points.ply", points_valid.numpy())

        mask = mask.reshape(grid_size, grid_size, grid_size).contiguous()

        # in x,y,z order
        # assumes that the scene is centered at the origin
        scene_box = SceneBox(
            aabb=torch.from_numpy(np.stack([bbx_min, bbx_max]) * scale),
            coarse_binary_gird=mask,
        )

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            width=widths,
            height=heights,
            camera_type=CameraType.PERSPECTIVE,
        )

        # indices = indices[::20]
        cameras = cameras[torch.tensor(indices)]
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices]
        semantic_filenames = [semantic_filenames[i] for i in indices]
        sparse_pts = [sparse_pts[i] for i in indices]

        # if self.config.include_mono_prior:
        #     depth_filenames = [depth_filenames[i] for i in indices]
        #     normal_filenames = [normal_filenames[i] for i in indices]
        # if self.config.include_sensor_depth:
        #     sensor_filenames = [sensor_filenames[i] for i in indices]

        metadata = {
            # "include_mono_prior": self.config.include_mono_prior,
            "sparse_pts": sparse_pts
        }

        if split == "train":
            metadata.update(
                self._load_3D_points(self.config.data, transform_matrix, scale, origin)
            )
        if self.config.include_semantics:
            classes = [id_label_mapping_ade20k[i] for i in range(len(id_label_mapping_ade20k))]
            semantics = Semantics(filenames=semantic_filenames, classes=classes, colors=None,
                                  mask_classes=transient_objects)
            metadata["semantics"] = semantics

        assert len(cameras) == len(image_filenames)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            mask_filenames=mask_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_transform=transform_matrix,
            dataparser_scale=scale,
            metadata=metadata,
        )

        return dataparser_outputs


    def _load_3D_points(
        self, colmap_path: Path, transform_matrix: torch.Tensor, scale_factor: float, origin: torch.Tensor
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
                    (points3D - origin) * scale_factor,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            ).float()
            @ transform_matrix.T
        )

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
                )
            out["points3D_image_ids"] = torch.stack(points3D_image_ids, dim=0)
            out["points3D_points2D_xy"] = torch.stack(points3D_image_xy, dim=0)
        return out

    def check_in_eval(self, idx):
        return idx in self.i_eval

    def find_eval_idx(self, idx):
        return self.i_eval[idx]


splatfactow_dataparser=DataParserSpecification(config=NerfWDataParserConfig())
