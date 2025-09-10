# ruff: noqa: E741
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

"""
Gaussian Splatting Model in the Wild implementation in nerfstudio.
https://kevinxu02.github.io/gsw.github.io/
"""

from gsplat.cuda._wrapper import spherical_harmonics
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from nerfstudio.data.dataparsers.heritage_dataparser import label_id_mapping_ade20k
from nerfstudio.data.utils.data_utils import compute_normals_finite_diff
from nerfstudio.utils import colormaps
from nerfstudio.cameras.camera_utils import normalize
from splatfactow.splatfactow_field import BGField, SplatfactoWField

import numpy as np
import torch
from torch import Tensor

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from pytorch_msssim import SSIM
from torch.nn import Parameter
import torch.nn.functional as F


from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE

def quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones(
        (1, 1, d, d), dtype=torch.float32, device=image.device
    )
    return (
        tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d)
        .squeeze(1)
        .permute(1, 2, 0)
    )


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class SplatfactoWModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: SplatfactoWModel)
    warmup_length: int = 1000
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 25
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 15000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 20000
    """stop splitting at this step"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(mode="off")
    )
    """Config of the camera optimizer to use"""
    enable_bg_model: bool = True
    """Whether to enable the 2d background model"""
    bg_num_layers: int = 3
    """Number of layers in the background model"""
    bg_layer_width: int = 128
    """Width of each layer in the background model"""
    implementation: Literal["tcnn", "torch"] = "torch"
    """Implementation of the models"""
    appearance_embed_dim: int = 48
    """Dimension of the appearance embedding, if 0, no appearance embedding is used"""
    app_num_layers: int = 3
    """Number of layers in the appearance model"""
    app_layer_width: int = 256
    """Width of each layer in the appearance model"""
    enable_alpha_loss: bool = True
    """Whether to enable the alpha loss for punishing gaussians from occupying background space, this also works with pure color background (i.e. white for overexposed skys)"""
    appearance_features_dim: int = 72
    """Dimension of the appearance feature"""
    enable_robust_mask: bool = True
    """Whether to enable robust mask for calculating the loss"""
    robust_mask_percentage: tuple = (0.0, 0.40)
    """The percentage of the entire image to mask out for robust loss calculation"""
    robust_mask_reset_interval: int = 6000
    """The interval to reset the mask"""
    never_mask_upper: float = 0.4
    """Whether to mask out the upper part of the image, which is usually the sky"""
    start_robust_mask_at: int = 6000
    """The step to start masking"""
    sh_degree_interval: int = 2000
    """The interval to increase the SH degree"""
    sh_degree: int = 3
    """The degree of SH to use for the color field"""
    bg_sh_degree: int = 4
    """The degree of SH to use for the background model"""
    use_avg_appearance: bool = False
    """Whether to use the average appearance embedding or 0-th for evaluation"""
    eval_right_half: bool = False
    """Whether to use the right half of the image for evluation"""
    depth_loss_mult: float = 0.0
    """Depth loss"""
    normal_loss_mult_l1: float = 0.0
    """Normal loss for l1 loss"""
    normal_loss_mult_cos: float = 0.0
    """Normal loss for cos loss"""
    depth_loss_disparity: bool = False
    """Calculate depth loss in disparity space (1/x)"""
    sky_loss_mult: float = 0.0
    """Sky loss"""
    ground_loss_mult: float = 0.0
    """Ground loss"""
    ground_depth_mult: float = 0.0
    """Ground depth multiplier"""
    color_loss: bool = False
    """Projecting mlp output to grayscale in rgb loss when input is grayscale"""


class SplatfactoWModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SplatfactoWModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        ground_classes = [
            "dirt track",
            "river", "sea", "mountain", "hill",
            "lamp",
            "field", "water", "earth", "grass",
            "floor",
            "land",
            "road", "path", "sidewalk",
            "sand", ]
        self.ground_indices = torch.tensor([label_id_mapping_ade20k[key] for key in ground_classes], dtype=torch.int64).view(1, 1, -1)
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter(
                (torch.rand((self.config.num_random, 3)) - 0.5)
                * self.config.random_scale
            )
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        appearance_features = torch.nn.Parameter(
            torch.zeros((num_points, self.config.appearance_features_dim))
            .float()
            .cuda()
        )
        if (
            self.seed_points is not None
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            # colors = torch.logit(self.seed_points[1] / 255, eps=1e-10).float().cuda()
            # rgb values of the seed points are in [0, 1] range
            colors = torch.nn.Parameter(self.seed_points[1] / 255).float().cuda()
        else:
            colors = torch.nn.Parameter(torch.zeros((num_points, 3))).cuda()

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "appearance_features": appearance_features,
                "colors": colors,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.max_loss = 0.0
        self.min_loss = 1e10

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        assert self.config.appearance_embed_dim > 0
        self.appearance_embeds = torch.nn.Embedding(
            self.num_train_data, self.config.appearance_embed_dim
        )

        if self.config.enable_bg_model:
            self.bg_model = BGField(
                appearance_embedding_dim=self.config.appearance_embed_dim,
                implementation=self.config.implementation,
                sh_levels=self.config.bg_sh_degree,
                num_layers=self.config.bg_num_layers,
                layer_width=self.config.bg_layer_width,
            )
        else:
            self.bg_model = None

        self.color_nn = SplatfactoWField(
            appearance_embed_dim=self.config.appearance_embed_dim,
            appearance_features_dim=self.config.appearance_features_dim,
            implementation=self.config.implementation,
            sh_levels=self.config.sh_degree,
            num_layers=self.config.app_num_layers,
            layer_width=self.config.app_layer_width,
        )

        self.cached_colors = None
        self.cached_bg_sh = None
        self.last_cam_idx = None
        self.cached_dirs = {}

        self.camera_idx = 0

    # def setup_shs(self, cam_idx: int):
    #     appearance_features = self.gauss_params["appearance_features"]
    #     appearance_embed = self.appearance_embeds(
    #         torch.tensor(cam_idx, device=self.device)
    #     )
    #     self.shs_0 =self.color_nn.shs_0(
    #     appearance_embed=appearance_embed.repeat(appearance_features.shape[0], 1),
    #     appearance_features=appearance_features,
    # )
    #     self.shs_rest = self.color_nn.shs_rest(
    #         appearance_embed=appearance_embed.repeat(appearance_features.shape[0], 1),
    #         appearance_features=appearance_features,
    #     )

    def set_camera_idx(self, cam_idx: int):
        self.camera_idx = cam_idx

    @property
    def shs_0(self):
        appearance_features = self.gauss_params["appearance_features"]
        appearance_embed = self.appearance_embeds(
            torch.tensor(self.camera_idx, device=self.device)
        )
        return self.color_nn.shs_0(
            appearance_embed=appearance_embed.repeat(appearance_features.shape[0], 1),
            appearance_features=appearance_features,
        )

    @property
    def shs_rest(self):
        appearance_features = self.gauss_params["appearance_features"]
        appearance_embed = self.appearance_embeds(
            torch.tensor(self.camera_idx, device=self.device)
        )
        return self.color_nn.shs_rest(
            appearance_embed=appearance_embed.repeat(appearance_features.shape[0], 1),
            appearance_features=appearance_features,
        )

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def appearance_features(self):
        return self.gauss_params["appearance_features"]

    @property
    def base_colors(self):
        return self.gauss_params["colors"]

    # @property
    # def features_rest(self):
    #     return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in [
                "means",
                "scales",
                "quats",
                "appearance_features",
                "opacities",
                "colors",
            ]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(
            n_neighbors=k + 1, algorithm="auto", metric="euclidean", n_jobs=None,
        ).fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(
                1 for _ in range(param_state["exp_avg"].dim() - 1)
            )
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(
                        *repeat_dims
                    ),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(
                        param_state["exp_avg_sq"][dup_mask.squeeze()]
                    ).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # type: ignore
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(
                    self.num_points, device=self.device, dtype=torch.float32
                )
                self.vis_counts = torch.ones(
                    self.num_points, device=self.device, dtype=torch.float32
                )
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads
            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval
                > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert (
                    self.xys_grad_norm is not None
                    and self.vis_counts is not None
                    and self.max_2Dsize is not None
                )
                avg_grad_norm = (
                    (self.xys_grad_norm / self.vis_counts)
                    * 0.5
                    * max(self.last_size[0], self.last_size[1])
                )
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (
                    self.scales.exp().max(dim=-1).values
                    > self.config.densify_size_thresh
                ).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (
                        self.max_2Dsize > self.config.split_screen_size
                    ).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (
                    self.scales.exp().max(dim=-1).values
                    <= self.config.densify_size_thresh
                ).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat(
                            [param.detach(), split_params[name], dup_params[name]],
                            dim=0,
                        )
                    )
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif (
                self.step >= self.config.stop_split_at
                and self.config.continue_cull_post_densification
            ):
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if (
                self.step < self.config.stop_split_at
                and self.step % reset_interval == self.config.refine_every
            ):
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(
                        torch.tensor(reset_value, device=self.device)
                    ).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (
            torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh
        ).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (
                torch.exp(self.scales).max(dim=-1).values
                > self.config.cull_scale_thresh
            ).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                if self.max_2Dsize is not None:
                    toobigs = (
                        toobigs
                        | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
                    )
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(
            f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}"
        )
        centered_samples = torch.randn(
            (samps * n_splits, 3), device=self.device
        )  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(
            dim=-1, keepdim=True
        )  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_appearance_features = self.appearance_features[split_mask].repeat(samps, 1)
        new_colors = self.base_colors[split_mask].repeat(samps, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(
            samps, 1
        )
        self.scales[split_mask] = torch.log(
            torch.exp(self.scales[split_mask]) / size_fac
        )
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "appearance_features": new_appearance_features,
            "colors": new_colors,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(
            f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}"
        )
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb
            )
        )
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "appearance_features", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        if self.config.enable_bg_model:
            assert self.bg_model is not None
            gps["field_background_encoder"] = list(self.bg_model.encoder.parameters())
            gps["field_background_base"] = list(self.bg_model.sh_base_head.parameters())
            gps["field_background_rest"] = list(self.bg_model.sh_rest_head.parameters())
        assert self.color_nn is not None
        assert self.appearance_embeds is not None
        gps["appearance_embed"] = list(self.appearance_embeds.parameters())
        gps["appearance_model_encoder"] = list(self.color_nn.encoder.parameters())
        gps["appearance_model_base"] = list(self.color_nn.sh_base_head.parameters())
        gps["appearance_model_rest"] = list(self.color_nn.sh_rest_head.parameters())

        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (
                    self.config.num_downscales
                    - self.step // self.config.resolution_schedule
                ),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(
        width: int, height: int, background: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "background": background,
        }

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
            bg_sh_degree_to_use = min(
                self.step // (self.config.sh_degree_interval // 2),
                self.config.bg_sh_degree,
            )

        # get the appearance embedding
        use_cached_sh = False
        if camera.metadata is not None and "cam_idx" in camera.metadata:
            cam_idx = camera.metadata["cam_idx"]
            if self.last_cam_idx is not None and not self.training:
                use_cached_sh = cam_idx == self.last_cam_idx and self.cached_colors.shape[0] == self.appearance_features.shape[0]
                if cam_idx != self.last_cam_idx:
                    CONSOLE.log("Current camera idx is", cam_idx)
            self.last_cam_idx = cam_idx
            appearance_embed = self.appearance_embeds(
                torch.tensor(cam_idx, device=self.device)
            )
        else:
            if self.config.use_avg_appearance:
                # calculate the average appearance embedding
                appearance_embed = self.appearance_embeds.weight.mean(dim=0)
            else:
                appearance_embed = self.appearance_embeds(
                    torch.tensor(0, device=self.device)
                )

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        opacities = self.opacities
        means = self.means
        appearance_features = self.appearance_features
        scales = self.scales
        quats = self.quats
        # base_colors = self.base_colors

        BLOCK_WIDTH = (
            16  # this controls the tile size of rasterization, 16 is a good default
        )
        K = camera.get_intrinsics_matrices().cuda()
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if use_cached_sh and self.cached_colors is not None:
            colors = self.cached_colors
        else:
            colors = self.color_nn(
                appearance_embed.repeat(appearance_features.shape[0], 1),
                appearance_features,
            ).float()
            self.cached_colors = colors

        render, alpha, info = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities).squeeze(-1),
            colors=colors,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        # background

        if self.config.enable_bg_model:
            # if (
            #     self.step > self.config.num_downscales * self.config.resolution_schedule
            #     and self.training
            # ):
            #     # cache directions in training, hope this won't cause memory issue
            #     if self.last_cam_idx not in self.cached_dirs:
            #         directions = normalize(
            #             camera.generate_rays(
            #                 camera_indices=0, keep_shape=False
            #             ).directions
            #         ).pin_memory()
            #         self.cached_dirs[self.last_cam_idx] = directions
            #     else:
            #         directions = self.cached_dirs[self.last_cam_idx]

            # else:
            directions = normalize(
                camera.generate_rays(camera_indices=0, keep_shape=False).directions
            )

            if use_cached_sh and not self.training and self.cached_bg_sh is not None:
                bg_sh_coeffs = self.cached_bg_sh
            else:
                bg_sh_coeffs = self.bg_model.get_sh_coeffs(
                    appearance_embedding=appearance_embed
                )
                self.cached_bg_sh = bg_sh_coeffs

            background = spherical_harmonics(
                degrees_to_use=bg_sh_degree_to_use,
                coeffs=bg_sh_coeffs.repeat(directions.shape[0], 1, 1),
                dirs=directions,
            )
            background = background.view(1, H, W, 3)
        else:
            background = self._get_background_color().view(1, 1, 1, 3)

        rgb = render[:, ..., :3] + (1 - alpha) * background

        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # depth image
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(
                alpha > 0, depth_im, depth_im.detach().max()
            ).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3:
            background = background.expand(H, W, 3)
        if render_mode == "RGB+ED":
            normals_bhw3, dilated_mask = compute_normals_finite_diff(
            depth_im[None, ..., 0], K,
            kernel_size=5,
            sigma=1.0)  # [-1, 1]

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im if render_mode == "RGB+ED" else None,  # type: ignore
 #           "normal": normals_bhw3.squeeze(0),  # type: ignore
 #           "normal_mask": dilated_mask.squeeze(0)[..., None],  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background.squeeze(0),  # type: ignore
        }  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        if self.config.color_loss:
            grayscale = self._downscale_if_required(batch["is_gray"])[:, :, 0] > 0.5
            rgb2gray = pred_img[grayscale][:, 0] * 0.2989 + \
                       pred_img[grayscale][:, 1] * 0.5870 + \
                       pred_img[grayscale][:, 2] * 0.1140
            pred_img[grayscale] = rgb2gray.unsqueeze(-1)
        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1_img = torch.abs(gt_img - pred_img)
        if (
            self.step >= self.config.start_robust_mask_at
            and self.config.enable_robust_mask
        ):
            robust_mask = self.robust_mask(Ll1_img)
            gt_img = gt_img * robust_mask
            pred_img = pred_img * robust_mask
            Ll1 = (Ll1_img * robust_mask).mean()
        else:
            Ll1 = Ll1_img.mean()

        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...], mask.permute(2, 0, 1)[None].tile(1, 3, 1, 1).bool()
        )
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
        # sky loss
        fg_mask_loss = torch.tensor(0.0).to(self.device)
        if "semantics" in batch and self.config.sky_loss_mult > 0:
            alpha = outputs["accumulation"]
            sky_mask = torch.round(self._downscale_if_required(batch["semantics"])) == 2
            sky_mask = sky_mask.to(self.device)
            if sky_mask.sum() != 0:
                fg_mask_loss = alpha[sky_mask].mean() * self.config.sky_loss_mult
            # sky loss
            # fg_label = (~sky_mask).float().to(self.device)  # sky
            # fg_mask_loss = F.l1_loss(alpha, fg_label) * self.config.sky_loss_mult

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1
            + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
            "sky_loss": fg_mask_loss,
        }
        if "sensor_depth" in batch and (self.config.depth_loss_mult > 0 or self.config.ground_depth_mult > 0):
            depths_gt = batch["sensor_depth"]

            # multiply certain classes
            depth_multiplier = torch.ones_like(mask) * self.config.depth_loss_mult
            if "semantics" in batch and self.config.ground_depth_mult > 0:
                ground_mask = torch.sum(batch["semantics"] == self.ground_indices, dim=-1, keepdim=True) != 0
                depth_multiplier[self._downscale_if_required(ground_mask).bool()] = self.config.ground_depth_mult

            depths_gt = self._downscale_if_required(depths_gt)
            depths_gt = depths_gt.to(self.device)
            if depths_gt.shape[-1] > 1:  # has confidence
                conf = depths_gt[:, :, 1:2]
                depths_gt = depths_gt[:, :, 0:1]
            else:  # no confidence
                conf = torch.ones_like(depths_gt).to(self.device)
                if "semantics" in batch and self.config.sky_loss_mult > 0:
                    conf *= ~sky_mask
            depths = outputs["depth"]
            if self.config.depth_loss_disparity:
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = torch.where(depths_gt > 0.0, 1.0 / depths_gt, torch.zeros_like(depths_gt))
                depthloss = torch.mean(F.l1_loss(disp, disp_gt, reduction="none") * conf * mask * depth_multiplier)
            else:
                depthloss = torch.mean(F.l1_loss(depths, depths_gt, reduction="none") * conf * mask * depth_multiplier)
            loss_dict["depth_loss"] = depthloss
        # normal loss
        if "normal_image" in batch and (self.config.normal_loss_mult_l1 > 0 or self.config.ground_depth_mult > 0):
            normal_pred = outputs["normal"]
            normal_mask = outputs["normal_mask"]
            normal_gt = batch["normal_image"]
            normal_gt = self._downscale_if_required(normal_gt).to(self.device)
            normal_gt[:, :, 0:3] = torch.nn.functional.normalize(normal_gt[:, :, 0:3], p=2, dim=0)
            # multiply certain classes
            depth_multiplier = torch.ones_like(mask) * (self.config.normal_loss_mult_l1 > 0).float()
            if "semantics" in batch and self.config.ground_depth_mult > 0:
                ground_mask = torch.sum(batch["semantics"] == self.ground_indices, dim=-1, keepdim=True) != 0
                depth_multiplier[self._downscale_if_required(ground_mask).bool()] = self.config.ground_depth_mult
            if normal_gt.shape[-1] > 3:  # has confidence
                normal_conf = normal_gt[:, :, 3:4]
                normal_gt = normal_gt[:, :, 0:3]
            else:
                normal_conf = torch.ones_like(normal_gt[..., :1]).to(self.device)
            normal_conf = normal_conf * normal_mask
            # if "semantics" in batch and self.config.sky_loss_mult > 0:
            #     normal_conf *= ~sky_mask
            normal_loss_l1 = torch.mean(F.l1_loss(normal_gt, normal_pred, reduction="none"), dim=-1, keepdim=True)
            normal_loss_cos = 1 - torch.sum(normal_gt * normal_pred, dim=-1, keepdim=True)
            loss_dict["normal_loss"] = torch.mean(mask * normal_conf * (normal_loss_l1 * depth_multiplier * self.config.normal_loss_mult_l1
                                                                        + normal_loss_cos * depth_multiplier * self.config.normal_loss_mult_cos))

        if self.config.enable_alpha_loss:
            alpha_loss = torch.tensor(0.0).to(self.device)
            background = outputs["background"]
            alpha = outputs["accumulation"]
            # for those pixel are well represented by bg and has low alpha, we encourage the gaussian to be transparent
            bg_mask = torch.abs(gt_img - background).mean(dim=-1, keepdim=True) < 0.003
            # use a box filter to avoid penalty high frequency parts
            f = 3
            window = (torch.ones((f, f)).view(1, 1, f, f) / (f * f)).cuda()
            bg_mask = (
                torch.nn.functional.conv2d(
                    bg_mask.float().unsqueeze(0).permute(0, 3, 1, 2),
                    window,
                    stride=1,
                    padding="same",
                )
                .permute(0, 2, 3, 1)
                .squeeze(0)
            )
            alpha_mask = bg_mask > 0.6
            # prevent NaN
            if alpha_mask.sum() != 0:
                alpha_loss = alpha[alpha_mask].mean() * 0.15
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
        loss_dict["alpha_loss"] = alpha_loss

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        predicted_rgb = outputs["rgb"]
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        acc = colormaps.apply_colormap(outputs["accumulation"])

        normal = None
  #      normal = outputs["normal"]
  #      normal = (normal + 1.0) / 2.0
  #      normal_mask = outputs["normal_mask"]

        combined_acc = torch.cat([acc], dim=1)

        # hacked version, only eval on the right half of the image
        # cut the image in half,HW3
        if self.config.eval_right_half:
            gt_rgb = gt_rgb[:, gt_rgb.shape[1] // 2 :, :]
            predicted_rgb = predicted_rgb[:, predicted_rgb.shape[1] // 2 :, :]

        # cv2.imwrite("gt.png", (gt_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
        # cv2.imwrite("p.png", (predicted_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
        # cv2.imwrite("c.png", (combined_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
        # exit()

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {}

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb, None)
        lpips = self.lpips(gt_rgb, predicted_rgb)
        metrics_dict.update({"psnr": float(psnr.item()),
                             "ssim": float(ssim),
                             "lpips": float(lpips),
                             })  # type: ignore
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            if self.config.eval_right_half:
                mask = mask[:, mask.shape[1] // 2 :, :]
            mask = mask.to(self.device)
            mask = mask.permute(2, 0, 1)[None].tile(1, 3, 1, 1).bool()

            gt_rgb = gt_rgb * mask
            predicted_rgb = predicted_rgb * mask
        else:
            mask = None

        psnr_black_mask = self.psnr(gt_rgb, predicted_rgb)
        ssim_black_mask = self.ssim(gt_rgb, predicted_rgb, None)
        ssim_masked = self.ssim(gt_rgb, predicted_rgb, mask)
        lpips_black_mask = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict.update({"psnr_black_mask": float(psnr_black_mask.item()),
                        "ssim_black_mask": float(ssim_black_mask),
                        "ssim_masked": float(ssim_masked),
                        "lpips_black_mask": float(lpips_black_mask),
                        })  # type: ignore

        # sky and transient mask by mult by 0
        sky_mask = torch.round(self._downscale_if_required(batch["semantics"])) != 2
        if self.config.eval_right_half:
            sky_mask = sky_mask.to(self.device)
            sky_mask = sky_mask[:, sky_mask.shape[1] // 2 :, :]
            sky_mask = sky_mask.permute(2, 0, 1)[None].tile(1, 3, 1, 1).bool()
        gt_rgb = gt_rgb * sky_mask
        predicted_rgb = predicted_rgb * sky_mask
        psnr_sky_mask = self.psnr(gt_rgb, predicted_rgb)
        ssim_sky_mask = self.ssim(gt_rgb, predicted_rgb, None)
        ssim_masked_sky = self.ssim(gt_rgb, predicted_rgb, mask*sky_mask)
        lpips_sky_mask = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict.update({"psnr_sky_mask": float(psnr_sky_mask.item()),
                        "ssim_sky_mask": float(ssim_sky_mask),
                        "ssim_masked_sky": float(ssim_masked_sky),
                        "lpips_sky_mask": float(lpips_sky_mask),
                        })  # type: ignore


        if "normal_image" in batch:
            normal_gt = batch["normal_image"].to(self.device)
            if normal_gt.shape[-1] > 3:  # has confidence
                normal_conf = normal_gt[:, :, 3:4]
                normal_gt = normal_gt[:, :, 0:3]
                normal_mask = torch.cat([normal_conf, normal_mask], dim=1)
            else:  # no confidence
                normal_conf = torch.ones_like(normal_gt[..., :1]).to(self.device)

            normal_gt = (normal_gt + 1.0) / 2.0
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        elif normal is not None:
            combined_normal = torch.cat([normal], dim=1)

        images_dict = {"img": combined_rgb,
                       "accumulation": combined_acc,
                #       "normal": combined_normal,
                #       "normal_mask": colormaps.apply_float_colormap(normal_mask),
                       "mask": colormaps.apply_float_colormap(mask[0].permute(1, 2, 0)),
                       }
        if "sensor_depth" in batch and (self.config.depth_loss_mult > 0 or self.config.ground_depth_mult > 0):
            depths_gt = batch["sensor_depth"]

            depths_gt = self._downscale_if_required(depths_gt)
            depths_gt = depths_gt.to(self.device)
            ground_mask = torch.ones_like(mask)
            if self.config.ground_depth_mult > 0:
                ground_mask = self._downscale_if_required(ground_mask)
            if depths_gt.shape[-1] > 1:  # has confidence
                conf = depths_gt[:, :, 1:2]
                depths_gt = depths_gt[:, :, 0:1]
                images_dict["depth_conf"] = colormaps.apply_float_colormap(conf)
            else:  # no confidence
                conf = torch.tensor(1.0).to(self.device)
            depth_pred = outputs["depth"]
            combined_depth = torch.cat([depths_gt * ground_mask, depth_pred], dim=1)
            combined_depth = colormaps.apply_depth_colormap(combined_depth)
            if "semantics" in batch:
                ground_mask = torch.sum(batch["semantics"] == self.ground_indices, dim=-1, keepdim=True) != 0
                images_dict["ground_mask"] = colormaps.apply_float_colormap(ground_mask.to(self.device))
            images_dict["depth"] = combined_depth
        elif "depth" in outputs:
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
            combined_depth = torch.cat([depth], dim=1)
            images_dict["depth"] = combined_depth


        images_dict["sky_mask"] = colormaps.apply_float_colormap(sky_mask.float()[0].permute(1, 2, 0))

        return metrics_dict, images_dict

    @torch.no_grad()
    def robust_mask(self, errors: torch.Tensor):
        """Computes Robust Mask.

        Args:
            errors: f32[h,w,c]. Per-subpixel errors.
            inlier_threshold: f32[]. Upper bound on per-pixel loss to use to determine
                if a pixel is an inlier or not.
            config: Config object.

        Returns:
            mask: f32[h,w,1].
        """
        epsilon = 1e-3
        # never mask the upper of the image
        errors[: int(errors.shape[0] * self.config.never_mask_upper), :, :] = 0.0
        Ll1 = errors.mean()
        # update max and min of Loss
        if (
            Ll1 > self.max_loss
            or self.step % self.config.robust_mask_reset_interval == 0
        ):
            self.max_loss = Ll1
        if Ll1 < self.min_loss:
            self.min_loss = Ll1

        mask_range_min, mask_range_max = self.config.robust_mask_percentage
        mask_percentage = (Ll1 - self.min_loss) / (
            (self.max_loss - self.min_loss) + 1e-6
        ) * (mask_range_max - mask_range_min) + mask_range_min

        errors = errors.view(1, errors.shape[0], errors.shape[1], errors.shape[2])
        error_per_pixel = torch.mean(errors, dim=-1, keepdim=True)
        inlier_threshold = torch.quantile(error_per_pixel, 1 - mask_percentage)
        mask = torch.ones_like(error_per_pixel)
        # 1.0 for inlier pixels.
        is_inlier_loss = (error_per_pixel <= inlier_threshold).float()
        # stats["is_inlier_loss"] = torch.mean(is_inlier_loss)

        # Apply 5x5 box filter.
        f = 5
        window = (torch.ones((f, f)).view(1, 1, f, f) / (f * f)).cuda()
        has_inlier_neighbors = torch.nn.functional.conv2d(
            is_inlier_loss.permute(0, 3, 1, 2), window, stride=1, padding="same"
        )
        has_inlier_neighbors = has_inlier_neighbors.permute(0, 2, 3, 1)  # [n,h,w,1]

        # Binarize after smoothing.
        has_inlier_neighbors = (has_inlier_neighbors > 0.4).float()

        # A pixel is an inlier if it is an inlier according to any of the above
        # criteria.
        mask = (
            (has_inlier_neighbors + is_inlier_loss > epsilon)
            .float()
            .view(errors.shape[1], errors.shape[2], 1)
        )

        del errors
        del has_inlier_neighbors
        del is_inlier_loss
        del error_per_pixel
        return mask

    @torch.no_grad()
    def render_equirect(self, W, appearance_embed=None):
        H = W // 2
        # For equirect, fx = fy = height = width/2
        fx, fy = torch.tensor(H), torch.tensor(H)
        cx, cy = torch.tensor(W / 2), torch.tensor(H / 2)
        # R = torch.eye(3, device=self.device)
        from nerfstudio.cameras.cameras import CameraType

        # flip the z and y axes to align with gsplat conventions
        # R = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        # combine R into c2w
        # c2w = torch.eye(4, device=self.device)
        # c2w[:3, :3] = R
        c2w = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            device=self.device,
        )
        c2w = c2w[None, :3, :]
        camera = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_worlds=c2w,
            camera_type=CameraType.EQUIRECTANGULAR,
        ).to(self.device)
        ray_bundle = camera.generate_rays(0, keep_shape=False, disable_distortion=True)
        assert self.bg_model is not None
        background = (
            self.bg_model(ray_bundle, appearance_embed)
            .float()
            .clamp(0, 1)
            .reshape(H, W, 3)
        )
        print("background shape: ", background.shape)
        print("background: ", background)
        # exit()
        self.save_image(background, "output_images/equirect_bg.jpg")
        return background

    # def save_image(self, img, path):
    #     import os

    #     import PIL

    #     img = img.detach().cpu().numpy()
    #     img = (img * 255).astype(np.uint8)
    #     img = PIL.Image.fromarray(img)
    #     # create output_images folder if it doesn't exist
    #     if not os.path.exists(path=os.path.dirname(path)):
    #         os.makedirs(os.path.dirname(path))
    #     img.save(path)
