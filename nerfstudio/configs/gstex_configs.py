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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.gstex import GStexModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

gstex_method_configs: Dict[str, TrainerConfig] = {}

gstex_method_configs["gstex-blender-init"] = TrainerConfig(
    method_name="gstex",
    steps_per_eval_image=0,
    steps_per_eval_batch=0,
    steps_per_save=0,
    steps_per_eval_all_images=0,
    max_num_iterations=1,
    mixed_precision=False,
    vis="viewer",
    gradient_accumulation_steps={"camera_opt": 100},
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=BlenderDataParserConfig(
            ),
            cache_images_type="uint8",
        ),
        model=GStexModelConfig(
            sh_degree=3,
            pixel_num=1e6,
            background_color="white",
            build_chart_every=100,
            fix_init=False,
            sigma_factor=3.0,
        ),
    ),
    optimizers={
        "xyz": {
            # scale by spatial_lr_scale which is around 5.0 for Blender [5.25, 5.118, etc.]
            "optimizer": AdamOptimizerConfig(lr=5 * 1.6e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5 * 1.6e-6,
                max_steps=15000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scaling": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None
        },
        "texture_dc": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
        },
    },
    viewer=ViewerConfig(
        num_rays_per_chunk=1 << 15,
    ),
)

gstex_method_configs["gstex-colmap-init"] = TrainerConfig(
    method_name="gstex",
    steps_per_eval_image=0,
    steps_per_eval_batch=0,
    steps_per_save=0,
    steps_per_eval_all_images=0,
    max_num_iterations=1,
    mixed_precision=False,
    vis="viewer",
    gradient_accumulation_steps={"camera_opt": 100},
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(
                load_3D_points=True,
                eval_mode="interval",
                eval_interval=8,
                orientation_method="none",
                center_method="none",
                auto_scale_poses=False,
                downscale_factor=2,
            ),
            cache_images_type="uint8",
        ),
        model=GStexModelConfig(
            sh_degree=3,
            pixel_num=1e7,
            background_color="black",
            build_chart_every=100,
            fix_init=True,
            sigma_factor=3.0,
        ),
    ),
    optimizers={
        "xyz": {
            # scale by spatial_lr_scale which is around 2.0 for DTU [1.85, 1.85, 1.784 etc.]
            "optimizer": AdamOptimizerConfig(lr=2 * 1.6e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=2 * 1.6e-6,
                max_steps=15000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scaling": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None
        },
        "texture_dc": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
        },
    },
    viewer=ViewerConfig(
        num_rays_per_chunk=1 << 15,
    ),
)

gstex_method_configs["gstex-blender-nvs"] = TrainerConfig(
    method_name="gstex",
    steps_per_eval_image=0,
    steps_per_eval_batch=0,
    steps_per_save=0,
    steps_per_eval_all_images=0,
    max_num_iterations=15000,
    mixed_precision=False,
    vis="viewer+tensorboard",
    gradient_accumulation_steps={"camera_opt": 100},
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=BlenderDataParserConfig(
            ),
            cache_images_type="uint8",
        ),
        model=GStexModelConfig(
            sh_degree=3,
            pixel_num=1e6,
            background_color="white",
            build_chart_every=100,
            fix_init=False,
            sigma_factor=3.0,
        ),
    ),
    optimizers={
        "xyz": {
            # scale by spatial_lr_scale which is around 5.0 for Blender [5.25, 5.118, etc.]
            "optimizer": AdamOptimizerConfig(lr=5 * 1.6e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5 * 1.6e-6,
                max_steps=15000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scaling": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None
        },
        "texture_dc": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
        },
    },
    viewer=ViewerConfig(
        num_rays_per_chunk=1 << 15,
        quit_on_train_completion=True,
    ),
)

gstex_method_configs["gstex-dtu-nvs"] = TrainerConfig(
    method_name="gstex",
    steps_per_eval_image=0,
    steps_per_eval_batch=0,
    steps_per_save=0,
    steps_per_eval_all_images=0,
    max_num_iterations=15000,
    mixed_precision=False,
    vis="viewer+tensorboard",
    gradient_accumulation_steps={"camera_opt": 100},
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(
                load_3D_points=True,
                eval_mode="interval",
                eval_interval=8,
                orientation_method="none",
                center_method="none",
                auto_scale_poses=False,
                downscale_factor=2,
            ),
            cache_images_type="uint8",
        ),
        model=GStexModelConfig(
            sh_degree=3,
            pixel_num=1e6,
            background_color="black",
            build_chart_every=100,
            fix_init=True,
            sigma_factor=3.0,
        ),
    ),
    optimizers={
        "xyz": {
            # scale by spatial_lr_scale which is around 2.0 for DTU [1.85, 1.85, 1.784 etc.]
            "optimizer": AdamOptimizerConfig(lr=2 * 1.6e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=2 * 1.6e-6,
                max_steps=15000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scaling": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None
        },
        "texture_dc": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
        },
    },
    viewer=ViewerConfig(
        num_rays_per_chunk=1 << 15,
        quit_on_train_completion=True,
    ),
)

gstex_method_configs["gstex-blender-lod"] = TrainerConfig(
    method_name="gstex",
    steps_per_eval_image=0,
    steps_per_eval_batch=0,
    steps_per_save=0,
    steps_per_eval_all_images=0,
    max_num_iterations=7000,
    mixed_precision=False,
    vis="viewer+tensorboard",
    gradient_accumulation_steps={"camera_opt": 100},
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=BlenderDataParserConfig(
            ),
            cache_images_type="uint8",
        ),
        model=GStexModelConfig(
            sh_degree=3,
            pixel_num=1e6,
            background_color="white",
            build_chart_every=100,
            fix_init=False,
            sigma_factor=3.0,
        ),
    ),
    optimizers={
        "xyz": {
            # scale by spatial_lr_scale which is around 5.0 for Blender [5.25, 5.118, etc.]
            "optimizer": AdamOptimizerConfig(lr=5 * 1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5 * 1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scaling": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None
        },
        "texture_dc": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
        },
    },
    viewer=ViewerConfig(
        num_rays_per_chunk=1 << 15,
        quit_on_train_completion=True,
    ),
)

gstex_method_configs["gstex-dtu-lod"] = TrainerConfig(
    method_name="gstex",
    steps_per_eval_image=0,
    steps_per_eval_batch=0,
    steps_per_save=0,
    steps_per_eval_all_images=0,
    max_num_iterations=7000,
    mixed_precision=False,
    vis="viewer+tensorboard",
    gradient_accumulation_steps={"camera_opt": 100},
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(
                load_3D_points=True,
                eval_mode="interval",
                eval_interval=8,
                orientation_method="none",
                center_method="none",
                auto_scale_poses=False,
                downscale_factor=2,
            ),
            cache_images_type="uint8",
        ),
        model=GStexModelConfig(
            sh_degree=3,
            pixel_num=1e6,
            background_color="black",
            build_chart_every=100,
            fix_init=True,
            fix_lod_init=True,
            sigma_factor=3.0,
        ),
    ),
    optimizers={
        "xyz": {
            # scale by spatial_lr_scale which is around 2.0 for DTU [1.85, 1.85, 1.784 etc.]
            "optimizer": AdamOptimizerConfig(lr=2 * 1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=2 * 1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scaling": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None
        },
        "texture_dc": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
        },
    },
    viewer=ViewerConfig(
        num_rays_per_chunk=1 << 15,
        quit_on_train_completion=True,
    ),
)