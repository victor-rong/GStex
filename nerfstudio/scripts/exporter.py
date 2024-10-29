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
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.models.gstex import GStexModel

from gstex_cuda.sh import num_sh_bases, spherical_harmonics

@dataclass
class Exporter:
    """Export the scene from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""

@dataclass
class ExportGStexPly(Exporter):
    """
    Export GStex model to a .ply
    Colors of each point are based on the average RGB across the Gaussian's texture
    """

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model: GStexModel = pipeline.model

        filename = self.output_dir / "splat.ply"

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            colors = torch.clamp(model.get_average_colors(), 0.0, 1.0).data.cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(str(filename), pcd)

class ExportGStexNpz(Exporter):
    """
    Export GStex model to a .npz
    """

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model: GStexModel = pipeline.model

        npz_filename = self.output_dir / "splat.npz"
        param_dict = {
            "xyz": model.means,
            "features_rest": model.features_rest,
            "opacity": model.opacities,
            "scaling": model.scales,
            "rotation": model.quats,
            "texture_dc": model.texture_dc.texture,
            "texture_dims": model.texture_dims,
            "mappings": model.mappings,
        }
        np_dict = {
            k: param_dict[k].detach().cpu().numpy() for k in param_dict
        }
        np.savez(
            npz_filename,
            **np_dict
        )

Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportGStexPly, tyro.conf.subcommand(name="gstex-ply")],
        Annotated[ExportGStexNpz, tyro.conf.subcommand(name="gstex-npz")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
