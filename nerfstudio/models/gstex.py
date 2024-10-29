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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from gstex_cuda._torch_impl import quat_to_rotmat
from gstex_cuda.texture import texture_gaussians
from gstex_cuda.texture_edit import texture_edit
from gstex_cuda.get_aabb_2d import get_aabb_2d, get_num_tiles_hit_2d, project_points
from gstex_cuda.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal
from plyfile import PlyData

from nerfstudio.cameras.cameras import Cameras, json_to_camera
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.utils.rotations import matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix


from nerfstudio.models.jagged_texture import JaggedTexture, texture_dims_to_query, texture_dims_to_int_coords

import open3d as o3d
from PIL import Image
import cv2
from nerfstudio.viewer.viewer_elements import ViewerRGB, ViewerSlider, ViewerButton, ViewerControl, ViewerClick

from datetime import datetime
from pathlib import Path
import os
import json
import subprocess

def get_formatted_time():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

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


def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

def depths_to_points(depths, viewmat, c2w, intrins, H, W):
    """
    Converts depth map to points
    """
    origin = torch.zeros((4, 1), device=c2w.device)
    origin[3] = 1
    origin = (c2w @ origin).squeeze(-1)

    image_jis = torch.stack(
        torch.meshgrid(torch.arange(0, H), torch.arange(0, W)),
        dim=-1
    ).to(device=c2w.device)
    image_jis = image_jis.reshape(-1, 2)

    fx, fy, cx, cy = intrins

    ndc_x = (image_jis[:,1] - cx + 0.5) / fx
    ndc_y = (image_jis[:,0] - cy + 0.5) / fy
    rays = torch.stack((ndc_x, ndc_y, torch.ones_like(ndc_x), torch.zeros_like(ndc_x)), dim=-1) @ c2w.T
    rays = rays / (torch.sqrt(torch.sum(rays**2, dim=-1, keepdim=True)) + 1e-9)
    view_rays = rays @ viewmat.T
    rays = rays.reshape(*depths.shape[:-1], -1)
    view_rays = view_rays.reshape(*depths.shape[:-1], -1)
    # don't use view depth
    ts = depths.squeeze(-1) / view_rays[...,2]
    samples = origin[...,:3] + ts[...,None] * rays[...,:3]
    samples = samples.reshape(H, W, -1)
    return samples

def depth_to_normal(depths, viewmat, c2w, intrins, H, W):
    """
    Estimates normal map from depths
    """
    points = depths_to_points(depths, viewmat, c2w, intrins, H, W)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

@dataclass
class GStexModelConfig(ModelConfig):
    """GStex Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: GStexModel)
    init_file: Optional[str] = None
    """initial file"""
    init_lod_ply: Optional[str] = None
    """initial lod ply file"""
    init_ply: Optional[str] = None
    """initial ply file"""
    init_npz: Optional[str] = None
    """initial npz file"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 2.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    settings: int = (1<<9) | (1<<10)
    """settings used by the CUDA rasterization. the default 1<<9 with 1<<10 corresponds to using the
    anti-aliasing blur and computing the NDC depth regularization as done in 2DGS.
    """
    lambda_normal: float = 0.0
    """loss coefficient for normal regularization as done in 2DGS"""
    lambda_reg: float = 0.0
    """loss coefficient for L2 depth distortion regularization as done in 2DGS"""
    build_chart_every: int = 100
    """how frequently to resize the texture dimensions"""
    pixel_num: float = 1e6
    """how many texels to have in total, generally with a .1% error. pixel_num can be set to 0 to
    simulate 2DGS behaviour."""
    use_normal_loss: bool = False
    """whether the normal regularization is used"""
    fix_init: bool = False
    """whether to fix the coordinate system of the input 2DGS (true for COLMAP datasets,
    false for NeRF synthetic datasets)"""
    fix_lod_init: bool = False
    """whether to fix the coordinate system of the input point cloud (true for COLMAP datasets,
    false for NeRF synthetic datasets)"""
    sigma_factor: float = 3.0
    """the per-Gaussian textures cover the region from [-sigma_factor * std, sigma_factor * std]
    across both axes"""
    import_edit_json: Optional[str] = None
    """allows user to copy over edit canvases across different models"""
    export_edit_dir: str = "./edits/"
    """where to save edit canvases"""

class GStexModel(Model):
    """Implementation of GStex

    Args:
        config: GStex configuration to instantiate model
    """

    config: GStexModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        self.measure_fps = False
        self.texture_set = False
        self.mapping_set = False
        self.fixed_texture = None
        self.setup_viewer()

        self.seed_other = None
        self.settings = self.config.settings
        self.register_buffer("pixel_scale", 10.0 * torch.ones(1, dtype=torch.float32))
        init_colors = None
        load_init = None
        if self.config.init_ply is not None:
            load_init = self.load_ply(str(self.config.init_ply))
            self.seed_points = [load_init[0], load_init[1]]
            scales = load_init[4]
            scales = torch.cat([scales, torch.ones_like(scales[:,:1])], dim=-1)
            self.seed_other = [
                load_init[3], scales, load_init[5]
            ]
        elif self.config.init_npz is not None:
            init_properties = np.load(self.config.init_npz, allow_pickle=True)
            init_means = torch.from_numpy(init_properties["xyz"])
            init_colors = torch.clamp(255.0 * torch.from_numpy(init_properties["colors"]), min=1.0, max=254.0)
            self.seed_points = [init_means, init_colors]
            self.seed_other = [
                torch.from_numpy(init_properties["opacity"]),
                torch.from_numpy(init_properties["scaling"]),
                torch.from_numpy(init_properties["rotation"]),
            ]
        elif self.config.init_lod_ply is not None:
            init_means, init_colors = self.load_from_lod_ply(self.config.init_lod_ply)
            self.seed_points = [init_means, init_colors]
        elif self.config.init_file is not None:
            init_means, init_colors = self.load_from_file(self.config.num_random, self.config.init_file)
            self.seed_points = [init_means, init_colors]
        
        if self.seed_points is not None and not self.config.random_init:
            self.means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)

        if (
            self.seed_other is not None
            and not self.config.random_init
        ):
            self.opacities = torch.nn.Parameter(self.seed_other[0])
            self.scales = torch.nn.Parameter(self.seed_other[1])
            self.quats = torch.nn.Parameter(self.seed_other[2])
            
        else:
            self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))
            self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
            self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))

        temp_mappings = torch.ones((self.num_points, 2))

        self.mappings = torch.nn.Parameter(temp_mappings)

        dim_sh = num_sh_bases(self.config.sh_degree)

        self.register_buffer("test_colors", torch.rand((self.num_points, 3), dtype=torch.float32))

        if load_init is not None:
            self.features_dc = torch.nn.Parameter(load_init[1][:,0,:])
            self.features_rest = torch.nn.Parameter(load_init[2])
        elif (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-5)
            self.features_dc = torch.nn.Parameter(shs[:, 0, :])
            self.features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            self.features_dc = torch.nn.Parameter(torch.rand(self.num_points, 3))
            self.features_rest = torch.nn.Parameter(torch.zeros((self.num_points, dim_sh - 1, 3)))

        self.texture_channels = 9 + 3 * dim_sh
        self.register_buffer(
            "texture_dims",
            torch.ones(self.num_points, 3, dtype=torch.int32)
        )
        hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        self.texture_dims[:,2] = torch.cumsum(hws, dim=0) - hws

        self.texture_dc = JaggedTexture(self.texture_dims, out_dim=3)
        self.texture_dc.texture.data = self.features_dc.data.clone().to(self.texture_dc.texture.device)

        self.step = 0

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        
        self.draw_camera = None
        self.edit_info = []
        if self.config.import_edit_json is not None:
            with open(self.config.import_edit_json, "r") as f:
                file_edit_info = json.load(f)
                for edit in file_edit_info:
                    camera_info = edit["camera"]
                    img_filename = edit["file"]
                    canvas = np.array(Image.open(img_filename))
                    self.edit_info.append({
                        "camera": camera_info,
                        "file": img_filename,
                        "canvas": canvas,
                    })

        self.build_charts()

    def setup_viewer(self):
        self.rerender_cb = lambda x : None
        self.edit_dir = Path(self.config.export_edit_dir) / get_formatted_time()
        self.lock_draw_camera = False
        self.edit_texture = None
        self.curve_ongoing = []
        
        self.cur_line_rgb = (0, 0, 0)
        self.cur_line_width = 5

        self.viewer_control = ViewerControl()

        self.n1_rgb_select = ViewerRGB(name="Polyline Colour", default_value=(0,0,0), cb_hook=self.select_rgb)
        self.n2_width_select = ViewerSlider(name="Polyline Width", default_value=5, min_value=1, max_value=20, step=1, cb_hook=self.select_width)
        self.n3_register_button = ViewerButton(name="Start Polyline", disabled=False, cb_hook=self.register_cb)
        self.n4_unregister_button = ViewerButton(name="End Polyline", disabled=True, cb_hook=self.unregister_cb)
        self.n5_reset_button = ViewerButton(name="Undo Polyline", disabled=True, cb_hook=self.handle_undo)
        self.n6_save_button = ViewerButton(name="Save Edit", cb_hook=self.handle_save)

    def set_rerender_cb(self, rerender_cb):
        self.rerender_cb = rerender_cb

    def handle_curve(self, button):
        self.curve_ongoing = []

    def handle_save(self, button):
        (self.edit_dir / "images").mkdir(parents=True, exist_ok=True)
        file_edit_info = []
        for edit in self.edit_info:
            camera_info = edit["camera"]
            img_filename = edit["file"]
            canvas = edit["canvas"]
            canvas_img = Image.fromarray(canvas)
            canvas_img.save(img_filename)
            file_edit_info.append({
                "camera": camera_info,
                "file": img_filename,
            })
        with open(os.path.join(self.edit_dir, "info.json"), "w") as f:
            json.dump(file_edit_info, f)
        self.cur_edit_img = None

    def handle_undo(self, button):
        if len(self.edit_info) <= 1:
            self.n5_reset_button.set_disabled(True)
        self.edit_info = self.edit_info[:-1]
        self.update_edit_texture()

    def update_edit_texture(self):
        edit_texture = torch.clone(SH2RGB(self.texture_dc.get_texture()))
        for i in range(len(self.edit_info)):
            camera_info = self.edit_info[i]["camera"]
            canvas = self.edit_info[i]["canvas"]
            change_img = canvas.astype(np.float32) / 255.0
            change_img = torch.from_numpy(change_img).to(device=self.means.device)
            camera = json_to_camera(camera_info).to(device=self.means.device)
            edit_texture = self.draw_from_view(camera, edit_texture, change_img)

        self.edit_texture = edit_texture

    # https://github.com/nerfstudio-project/viser/pull/157
    def pointer_click_cb(self, click: ViewerClick):
        h = self.edit_info[-1]["canvas"].shape[0]
        w = self.edit_info[-1]["canvas"].shape[1]
        xy = click.screen_pos
        self.curve_ongoing.append((int(w * xy[0]), int(h * xy[1])))
        if len(self.curve_ongoing) > 1:
            self.edit_info[-1]["canvas"] = self.draw_edit_line(self.curve_ongoing[-2:], self.edit_info[-1]["canvas"])
        self.update_edit_texture()
        self.rerender_cb()


    def register_cb(self, button: ViewerButton):
        self.n3_register_button.set_disabled(True)
        self.n4_unregister_button.set_disabled(False)
        self.n5_reset_button.set_disabled(True)
        self.lock_draw_camera = True

        camera = self.draw_camera.to_json(0)
        save_file = self.edit_dir / "images" / f"{get_formatted_time()}.png"
        canvas = np.zeros((self.draw_camera.height.item(), self.draw_camera.width.item(), 4), np.uint8)
        self.edit_info.append({
            "camera": camera,
            "file": str(save_file),
            "canvas": canvas,
        })
        self.viewer_control.register_pointer_cb("click", self.pointer_click_cb)
    
    def unregister_cb(self, button: ViewerButton):
        self.n4_unregister_button.set_disabled(True)
        self.n3_register_button.set_disabled(False)
        self.curve_ongoing = []
        self.n5_reset_button.set_disabled(False)
        self.lock_draw_camera = False
        self.viewer_control.unregister_pointer_cb()
        self.update_edit_texture()
        self.rerender_cb()

    def select_rgb(self, element):
        self.cur_line_rgb = element.value
    
    def select_width(self, element):
        self.cur_line_width = element.value
    
    def draw_edit_line(self, lines, canvas):
        pts = np.array(lines, dtype=np.int32)
        color = self.cur_line_rgb + (255,)
        new_canvas = cv2.polylines(canvas, [pts], False, color, thickness=self.cur_line_width)
        return new_canvas

    def draw_from_view(self, camera, cur_texture, change_img):
        assert change_img.shape[-1] == 4
        assert isinstance(camera, Cameras)
        assert camera.shape[0] == 1, "Only one camera at a time"
        
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.means.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        c2w = viewmat.squeeze().inverse()

        # calculate the FOV of the camera given fx and fy, width and height
        fx = camera.fx.item()
        fy = camera.fy.item()
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        intrinsics = (fx, fy, cx, cy)

        means = torch.zeros_like(self.means)
        means[...] = self.means[...]

        quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        scales = torch.zeros_like(self.scales)
        scales[:,:-1] = torch.clamp(torch.exp(self.scales[:,:-1]), min=1e-9)
        scales[:,-1] = 1e-5 * torch.mean(scales[:,:-1], dim=-1).detach()
        opacities = torch.sigmoid(self.opacities)

        uv0, umap, vmap = self.get_uv_mapping(means, quats, self.mappings)

        BLOCK_WIDTH = 16 # this controls the tile size of rasterization, 16 is a good default

        xys, depths = project_points(means, viewmat.squeeze()[:3,:], intrinsics)

        centers, extents = get_aabb_2d(means, scales, 1, quats, viewmat.squeeze()[:3,:], intrinsics)

        num_tiles_hit = get_num_tiles_hit_2d(centers, extents, H, W, BLOCK_WIDTH)

        texture_info = (self.means.shape[0], 1, 3)
        texture_dims = self.texture_dims

        temp_outputs = texture_gaussians(
            texture_info,
            texture_dims,
            centers,
            extents,
            depths,
            num_tiles_hit,
            torch.sigmoid(means),
            opacities,
            means,
            scales,
            1,
            quats,
            uv0,
            umap,
            vmap,
            cur_texture,
            viewmat.squeeze()[:3, :],
            c2w,
            fx,
            fy,
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
            self.settings,
            background=torch.zeros(3, device=self.device),
            use_torch_impl=False,
        )
        depth_output = temp_outputs[1]
        depth_lower = depth_output - 1e-2
        depth_upper = depth_output + 1e-2

        updated_texture = texture_edit(
            (self.means.shape[0], 1, 5),
            texture_dims,
            change_img[:,:,:3],
            change_img[:,:,3:],
            depth_lower,
            depth_upper,
            centers,
            extents,
            depths,
            num_tiles_hit,
            opacities,
            means,
            scales,
            1,
            quats,
            uv0,
            umap,
            vmap,
            viewmat.squeeze()[:3, :],
            c2w,
            fx,
            fy,
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
            1<<13,
            background=torch.zeros(3, device=self.device),
            use_torch_impl=False,
        )
        weight = updated_texture[:,3:4] / (updated_texture[:,4:] + 1e-6)
        edit_rgb_texture = updated_texture[:,:3] / (updated_texture[:,3:4] + 1e-6)
        updated_texture = edit_rgb_texture * weight + cur_texture * (1 - weight)
        return updated_texture

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.config.sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.config.sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.tensor(xyz, dtype=torch.float).requires_grad_(True)
        features_dc = torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous()
        features_rest = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous()
        opacity = torch.tensor(opacities, dtype=torch.float)
        scaling = torch.tensor(scales, dtype=torch.float)
        rotation = torch.tensor(rots, dtype=torch.float)

        if self.config.fix_init:
            new_xyz = torch.zeros_like(xyz)
            new_xyz[:,0] = xyz[:,0]
            new_xyz[:,1] = xyz[:,2]
            new_xyz[:,2] = -xyz[:,1]

            rotmats = quaternion_to_matrix(rotation)
            rotmats_fix = torch.zeros_like(rotmats)
            rotmats_fix[:,0,:] = rotmats[:,0,:]
            rotmats_fix[:,1,:] = rotmats[:,2,:]
            rotmats_fix[:,2,:] = -rotmats[:,1,:,]
            new_rotations = matrix_to_quaternion(rotmats_fix)
        else:
            new_xyz = xyz
            new_rotations = rotation
        return [new_xyz, features_dc, features_rest, opacity, scaling, new_rotations]

    def load_draw_camera(self, camera):
        if self.lock_draw_camera:
            return
        self.draw_camera = camera

    def load_from_lod_ply(self, lod_ply_file: str):
        plydata = PlyData.read(lod_ply_file)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        rgbs = np.stack((np.asarray(plydata.elements[0]["red"]),
                        np.asarray(plydata.elements[0]["green"]),
                        np.asarray(plydata.elements[0]["blue"])),  axis=1)

        if self.config.fix_lod_init:
            new_xyz = np.zeros_like(xyz)
            new_xyz[:,0] = xyz[:,0]
            new_xyz[:,1] = xyz[:,2]
            new_xyz[:,2] = -xyz[:,1]
        else:
            new_xyz = np.zeros_like(xyz)
            new_xyz[:,0] = xyz[:,0]
            new_xyz[:,1] = xyz[:,1]
            new_xyz[:,2] = xyz[:,2]

        return torch.Tensor(new_xyz), torch.Tensor(rgbs)


    def load_from_file(self, num_points: int, pcd_file: str):
        pcd = o3d.io.read_point_cloud(pcd_file)
        orig_points = np.array(pcd.points).astype(np.float32)
        orig_colors = np.array(pcd.colors).astype(np.float32)
        indices = list(range(orig_points.shape[0]))
        np.random.shuffle(indices)

        if num_points == -1:
            num_points = orig_points.shape[0]
        points = torch.Tensor(
            orig_points[indices]
        )[:num_points,:]
        colors = 255.0 * torch.Tensor(
            orig_colors[indices]
        )[:num_points,:]
        return points, colors

    def get_average_colors(self):
        idxs = torch.arange(self.texture_dims.shape[0], dtype=torch.int64, device=self.texture_dims.device)
        hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        ids = torch.repeat_interleave(idxs, hws, dim=0)
        texture_dc = self.texture_dc.get_texture()
        if self.config.sh_degree > 0:
            texture_dc = SH2RGB(texture_dc)
        else:
            texture_dc = torch.sigmoid(texture_dc)
        denom = hws.float()
        avg_dc = torch.zeros((denom.shape[0], texture_dc.shape[-1],), device=hws.device, dtype=torch.float32)
        avg_dc = torch.index_add(avg_dc, 0, ids, texture_dc) / denom[:,None]
        return avg_dc

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    def load_state_dict(self, dict, **kwargs):
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        self.mappings = torch.nn.Parameter(torch.zeros(newp, 2, device=self.device))
        self.features_dc = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.features_rest = torch.nn.Parameter(
            torch.zeros(newp, num_sh_bases(self.config.sh_degree) - 1, 3, device=self.device)
        )
        self.texture_dims = torch.zeros(newp, 3, dtype=torch.int32, device=self.device)
        self.test_colors = torch.zeros(newp, 3, dtype=torch.float32, device=self.device)
        
        rand_colors = torch.ones((self.num_points, 3), dtype=torch.float32, device=self.device)
        rand_weight = torch.rand((self.num_points,), dtype=torch.float32, device=self.device)
        rand_colors[:,0] = 0.5 + 0.5 * rand_weight
        rand_colors[:,2] = 0.5 + 0.5 * rand_weight
        
        self.texture_dc.init_from_dims(dict["texture_dims"], initial=True)
        new_size = dict["texture_dc.texture"].shape[0]
        self.texture_dc.adjust_texture_size(new_size)

        super().load_state_dict(dict, **kwargs)

        self.test_colors = rand_colors
        self.texture_dc.init_from_dims(self.texture_dims)
        self.edit_texture = None
        self.update_edit_texture()

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

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
    
    def flush(self, step: int):
        assert step == self.step
        torch.cuda.empty_cache()

    def reshape_in_optim(self, optimizer, new_params):
        if len(new_params) != 1:
            return
        assert len(new_params) == 1
        assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]

        if param in optimizer.state:
            param_state = optimizer.state[param]
            del optimizer.state[param]
            
            if "exp_avg" in param_state:
                param_state["exp_avg"] = torch.zeros_like(new_params[0].data)
            if "exp_avg_sq" in param_state:
                param_state["exp_avg_sq"] = torch.zeros_like(new_params[0].data)
            
            del optimizer.param_groups[0]["params"][0]
            del optimizer.param_groups[0]["params"]
            optimizer.param_groups[0]["params"] = new_params

            optimizer.state[new_params[0]] = param_state

        else:
            del optimizer.param_groups[0]["params"][0]
            del optimizer.param_groups[0]["params"]
            optimizer.param_groups[0]["params"] = new_params

    def reshape_in_all_optim(self, optimizers):
        param_groups = self.get_texture_param_groups()
        for group, param in param_groups.items():
            self.reshape_in_optim(optimizers.optimizers[group], param)
        torch.cuda.empty_cache()

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color
    
    def build_charts(self, update_pixel_scale=True):
        total_pixel_num = self.config.pixel_num
        with torch.no_grad():
            sigma_factor = self.config.sigma_factor
            length0 = torch.exp(self.scales[:,0])
            length1 = torch.exp(self.scales[:,1])
            def get_score(x):
                score = torch.sum(torch.ceil(sigma_factor * length0 / x) * torch.ceil(sigma_factor * length1 / x)).item()
                return score

            self.pixel_scales = torch.ones_like(length0)
            adjustments = torch.ones_like(length0)            
            adjustments = torch.sqrt(adjustments**2 / torch.mean(adjustments**2))

            if update_pixel_scale:
                lo = 10.0
                hi = np.sqrt(torch.sum(sigma_factor * sigma_factor * length0 * length1 * (adjustments ** 2)).item() / total_pixel_num)
                mid = 0.5 * (lo + hi)
                score = get_score(mid)
                iter_num = 0
                tol = 1e-3
                while score < (1-tol) * total_pixel_num or score > (1+tol) * total_pixel_num:
                    if score < (1-tol) * total_pixel_num:
                        lo = mid
                    else:
                        hi = mid
                    mid = 0.5 * (lo + hi)
                    score = get_score(mid / adjustments)
                    iter_num += 1
                    if iter_num > 30:
                        break
                self.pixel_scale[0] = mid
                self.pixel_scales = mid / adjustments


            self.texture_dims = torch.zeros(self.num_points, 3, dtype=self.texture_dims.dtype, device=self.texture_dims.device)
            self.texture_dims[:,0] = torch.ceil(sigma_factor * length0 / self.pixel_scales)
            self.texture_dims[:,1] = torch.ceil(sigma_factor * length1 / self.pixel_scales)
            hws = self.texture_dims[:,0] * self.texture_dims[:,1]
            self.texture_dims[:,2] = torch.cumsum(hws, dim=0) - hws

            self.mappings[:,0] = 1 / (2.0 * sigma_factor * length0)
            self.mappings[:,1] = 1 / (2.0 * sigma_factor * length1)

        
            self.texture_dc.init_from_dims(self.texture_dims)
            self.edit_texture = None
            self.update_edit_texture()

    def retexture_after(self, optimizers: Optimizers, step):
        assert step == self.step
        with torch.no_grad():
            self.build_charts()
            self.reshape_in_all_optim(optimizers)
            self.flush(step)

    @property
    def num_points(self):
        return self.means.shape[0]

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))

        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.retexture_after,
                update_every_num_iters=self.config.build_chart_every,
                args=[training_callback_attributes.optimizers],
            )
        )

        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.flush,
                update_every_num_iters=500,
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        params = {
            "xyz": [self.means],
            "features_dc": [self.features_dc],
            "features_rest": [self.features_rest],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
        }
        return params

    def get_texture_param_groups(self) -> Dict[str, List[Parameter]]:
        params = {
            "texture_dc": [self.texture_dc.texture],
        }
        return params

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        gps.update(self.get_texture_param_groups())
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def get_uv_mapping(self, means, quats, mappings):
        uv0 = 0.5 * torch.ones_like(means[:,:2])
        Rs = quat_to_rotmat(quats.detach())
        ax1 = Rs[:,:,0].detach() # important!! umap and vmap are in terms of axes but that's more of an unfortunate accident than anything
        ax2 = Rs[:,:,1].detach()
        ax3 = Rs[:,:,2].detach()


        umap = mappings[:,0,None].detach() * ax1
        vmap = mappings[:,1,None].detach() * ax2

        uv0 = uv0.unsqueeze(1)
        umap = umap.unsqueeze(1)
        vmap = vmap.unsqueeze(1)
        
        return uv0, umap, vmap

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

        extra_stuff = (not self.measure_fps) and (not self.training)
        
        assert isinstance(camera, Cameras)
        assert camera.shape[0] == 1, "Only one camera at a time"

        if not self.training:
            self.load_draw_camera(camera)
        
        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if extra_stuff:
            camera_downscale = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        c2w = viewmat.squeeze().inverse()

        # calculate the FOV of the camera given fx and fy, width and height
        fx = camera.fx.item()
        fy = camera.fy.item()
        cx = camera.cx.item()
        cy = camera.cy.item()
        self.intrinsics = (fx, fy, cx, cy)
        fovx = 2 * math.atan(camera.width / (2 * fx))
        fovy = 2 * math.atan(camera.height / (2 * fy))
        W, H = int(camera.width.item()), int(camera.height.item())
        if extra_stuff:
            self.last_size = (H, W)
            projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
            self.proj = projmat @ viewmat
        self.viewmat = viewmat

        means = torch.zeros_like(self.means)
        means[...] = self.means[...]

        quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        scales = torch.zeros_like(self.scales)
        scales[:,:-1] = torch.clamp(torch.exp(self.scales[:,:-1]), min=1e-9)
        scales[:,-1] = 1e-5 * torch.mean(scales[:,:-1], dim=-1).detach()
        opacities = torch.sigmoid(self.opacities)

        if self.training or extra_stuff:
            uv0, umap, vmap = self.get_uv_mapping(means, quats, self.mappings)
        else:
            if not self.mapping_set:
                self.mapping_set = True
                self.uv0, self.umap, self.vmap = self.get_uv_mapping(means, quats, self.mappings)

        BLOCK_WIDTH = 16 # this controls the tile size of rasterization, 16 is a good default

        xys, depths = project_points(means, viewmat.squeeze()[:3,:], self.intrinsics)

        centers, extents = get_aabb_2d(means, scales, 1, quats, viewmat.squeeze()[:3,:], self.intrinsics)
        num_tiles_hit = get_num_tiles_hit_2d(centers, extents, H, W, BLOCK_WIDTH)

        if extra_stuff:
            # rescale the camera back to original dimensions before returning
            camera.rescale_output_resolution(camera_downscale)

        texture_channels = 3
        if extra_stuff:
            texture_channels = 6
        
        total_size = self.texture_dc.total_size

        if self.training or extra_stuff:
            texture = torch.zeros((total_size, texture_channels), device=self.means.device)
            texture_dc = self.texture_dc.get_texture()

        texture_dims = self.texture_dims
        texture_info = (self.num_points, 1, texture_channels)

        if self.config.sh_degree > 0:
            colors = torch.cat((torch.zeros_like(self.features_dc[:, None, :]), self.features_rest), dim=1)
            viewdirs = means.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            if self.config.fix_init:
                new_viewdirs = torch.zeros_like(viewdirs)
                new_viewdirs[:,0] = viewdirs[:,0]
                new_viewdirs[:,1] = -viewdirs[:,2]
                new_viewdirs[:,2] = viewdirs[:,1]
                rgbs = spherical_harmonics(n, new_viewdirs, colors)
            else:
                rgbs = spherical_harmonics(n, viewdirs, colors)
            gaussian_rgbs = rgbs
        else:
            gaussian_rgbs = torch.sigmoid(self.features_dc[:,:])
            texture[:,0:3] = torch.sigmoid(texture_dc)

        if self.training or extra_stuff:
            if self.config.sh_degree > 0:
                texture[:,0:3] = SH2RGB(texture_dc)
            else:
                texture[:,0:3] = torch.sigmoid(texture_dc)
        else:
            if not self.texture_set:
                self.texture_set = True
                self.fixed_texture = SH2RGB(self.texture_dc.get_texture())

        if not (num_tiles_hit > 0).any():
            rgb = torch.ones((H, W, 3), dtype=torch.float32, device=self.means.device)
            depth = torch.ones((H, W, 1), dtype=torch.float32, device=self.means.device)
            return {"rgb": rgb, "depth": depth}

        def custom_texture_gaussians(custom_rgbs, custom_texture, custom_opacities, custom_settings):
            custom_outputs = texture_gaussians(
                texture_info,
                texture_dims,
                centers,
                extents,
                depths,
                num_tiles_hit,
                custom_rgbs,
                custom_opacities,
                means,
                scales,
                1,
                quats,
                self.uv0 if self.mapping_set else uv0,
                self.umap if self.mapping_set else umap,
                self.vmap if self.mapping_set else vmap,
                custom_texture,
                viewmat.squeeze()[:3, :],
                c2w,
                fx,
                fy,
                cx,
                cy,
                H,
                W,
                BLOCK_WIDTH,
                custom_settings,
                background=torch.zeros_like(background),
                use_torch_impl=False,
            )
            return custom_outputs

        all_outputs = custom_texture_gaussians(
            gaussian_rgbs,
            self.fixed_texture if self.texture_set else texture,
            opacities,
            self.settings,
        )
        
        out_img, out_depth, out_reg, out_alpha, out_texture, out_normal = all_outputs[:6]

        if extra_stuff:
            test_img = torch.zeros_like(out_img)
            edit_img = torch.zeros_like(out_img)
            running_img = torch.zeros_like(out_img)
            clean_normal_img = torch.zeros_like(out_img)
            uv_im = torch.zeros_like(out_img)

        if extra_stuff:
            test_rgbs = self.test_colors
            updated_texture = torch.zeros_like(texture)
            updated_texture[:,3:] = texture[:,3:]
            if self.edit_texture is not None:
                updated_texture[:,:3] = self.edit_texture
            
            test_opacities = opacities.clone()
            test_opacities[test_opacities <= 0.5] = 0.0
            test_opacities[test_opacities > 0.2] = 1.0
            custom_settings = self.settings #| (1<<16) | (16<<17) | (4<<26) | (1<<25) | (1<<2)
            test_outputs = custom_texture_gaussians(test_rgbs, updated_texture, test_opacities, custom_settings)

            test_img = test_outputs[0] + (1 - test_outputs[3][:,:,None]) * background[None,None,:]
            uv_im = test_outputs[4][:,:,3:6] + (1 - test_outputs[3][:,:,None]) * background[None,None,:]
            uv_im = torch.clamp(uv_im, min=0.0, max=1.0)

            test_outputs_normal = custom_texture_gaussians(test_rgbs, updated_texture, opacities, self.settings | (1<<15))
            edit_img = out_img + test_outputs_normal[4][:,:,:3] + (1 - out_alpha[:,:,None]) * background[None,None,:]
            edit_img = torch.clamp(edit_img, min=0.0, max=1.0)
            clean_normal_img = 0.5 * (test_outputs_normal[5] + 1) + (1 - out_alpha[:,:,None]) * background[None,None,:]
            clean_normal_img = torch.clamp(clean_normal_img, min=0.0, max=1.0)

        rgb = out_img + out_texture[:,:,0:3] + (1 - out_alpha[:,:,None]) * background[None,None,:]
        rgb = torch.clamp(rgb, min=0.0, max=1.0)

        images = {}
        if not self.training and not extra_stuff:
            images = {"rgb": rgb, "background": background}
            return images

        depth = out_depth[...,None]
        alpha = out_alpha[...,None]
        reg = out_reg[...,None]
        normal_im = out_normal
        texture_rgbs = out_texture[:,:,0:3]

        if self.config.use_normal_loss:
            # detach gradient between estimated normal and depth
            estimated_normal_im = depth_to_normal(depth, viewmat, c2w, self.intrinsics, H, W).detach()
        else:
            estimated_normal_im = normal_im

        if self.training:
            images = {
                "rgb": rgb, "background": background, "depth": depth, "accumulation": alpha,
                "normal_im": normal_im, "estimated_normals": estimated_normal_im, "reg": reg
            }
            return images

        images.update({"rgb": rgb, "depth": depth, "accumulation": alpha, "test": test_img, "edit": edit_img,
            "clean_normal_img": clean_normal_img,
            "only_rgb": torch.clamp(out_img + 0.5, 0.0, 1.0), "only_texture": torch.clamp(texture_rgbs, 0.0, 1.0),
            "uv": uv_im, "running": running_img,
            "normal_im": normal_im, "estimated_normals": estimated_normal_im, "reg": reg, "background": background})
        return images

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
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        alpha_img = outputs["accumulation"]
        normal_img = outputs["normal_im"]
        estimated_normal_img = outputs["estimated_normals"]
        pred_img = outputs["rgb"]
        reg_img = outputs["reg"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        
        def get_val(config_vals):
            if isinstance(config_vals, float):
                return config_vals
            if isinstance(config_vals, int):
                return float(config_vals)
            if self.step >= config_vals[2]:
                return config_vals[1]
            return config_vals[0]

        lambda_reg = get_val(self.config.lambda_reg)
        lambda_normal = get_val(self.config.lambda_normal)

        normal_loss = lambda_normal * torch.mean(alpha_img.squeeze(-1) - torch.sum(normal_img * estimated_normal_img, dim=-1))
        reg_loss = lambda_reg * torch.mean(reg_img)
        return {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "normal_loss": normal_loss,
            "reg_loss": reg_loss,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
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
        extra_stuff = not self.measure_fps and not self.training
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            def n_im(im):
                return (im - im.min().item()) / (im.max().item() - im.min().item() + 1e-6)
            predicted_rgb = outputs["rgb"]

            if extra_stuff:
                predicted_depth = n_im(outputs["depth"].repeat(1, 1, 3))
                predicted_normals = outputs["clean_normal_img"]
                predicted_test = outputs["test"]
                predicted_uv = outputs["uv"]
                predicted_edit = outputs["edit"]

        if d >= 1:
            combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            gt_rgb_reshaped = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
            predicted_rgb_reshaped = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

            # apply uint8 quantization, nerfstudio does not do this by default
            predicted_rgb_reshaped = (255.0 * predicted_rgb_reshaped).to(dtype=torch.uint8)
            predicted_rgb_reshaped = predicted_rgb_reshaped.to(dtype=torch.float32) / 255.0
            
            psnr = self.psnr(gt_rgb_reshaped, predicted_rgb_reshaped)
            ssim = self.ssim(gt_rgb_reshaped, predicted_rgb_reshaped)
            lpips = self.lpips(gt_rgb_reshaped, predicted_rgb_reshaped)

            # all of these metrics will be logged as scalars
            metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
            metrics_dict["lpips"] = float(lpips)
        else:
            combined_rgb = predicted_rgb
            metrics_dict = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        metrics_dict["gaussian_count"] = float(self.means.shape[0])
        metrics_dict["texel_count"] = float(self.texture_dc.total_size)
        metrics_dict["pixel_scale"] = float(self.pixel_scale.item())

        if extra_stuff:
            images_dict = {"img": combined_rgb, "depth": predicted_depth, "normal": predicted_normals,
            "test": predicted_test, "uv": predicted_uv, "edit": predicted_edit}
        else:
            images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "rgb") -> torch.Tensor:
        """Returns the RGBA image from the outputs of the model.

        Args:
            outputs: Outputs of the model.

        Returns:
            RGBA image.
        """
        rgb = outputs[output_name]
        acc = outputs["accumulation"]
        if acc.dim() < rgb.dim():
            acc = acc.unsqueeze(-1)
        return torch.cat((rgb, acc), dim=-1)