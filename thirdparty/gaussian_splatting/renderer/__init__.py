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

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE ####################################
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################




import torch
import math
import time 
import torch.nn.functional as F
import time 

from scene.saro_gaussian import GaussianModel


from utils.sh_utils import eval_sh
from utils.graphics_utils import getProjectionMatrixCV, focal2fov, fov2focal

from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings, GaussianRasterizer




def test_render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, require_segment = False):
    """
    Render the scene. 
    基本完全参照gs-flow的实现
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means.这个东西只是用来记录梯度的，自己的值为0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # print(torch.cuda.is_available())
    # startime = time.time()
    torch.cuda.synchronize()

    # pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # bg_color = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
    # print(bg_color)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        # viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        # projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        # campos=viewpoint_camera.camera_center,
        prefiltered=False
        # ,debug=pipe.debug
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    means2D = screenspace_points

    cov3D_precomp = None

    colors_precomp = None
    startime = time.time()

    means3D,rotations,scales,opacity,shs,_ = pc.get_deformation_eval(viewpoint_camera.timestamp)
    rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # print(torch.cuda.is_available())

    torch.cuda.synchronize()
    duration = time.time() - startime
    res_dic = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration": duration
            }
    if require_segment:
        #The time taken to render segments should not be recorded in the rendering time.
        means3D,rotations,scales,opacity,shs,_ = pc.get_deformation(viewpoint_camera.timestamp)
        colors_precomp = pc.get_trbfscale.detach().expand(-1,3) 
        shs = None
        rendered_image, radii,depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
        res_dic["segment_render"] = rendered_image

    return res_dic
