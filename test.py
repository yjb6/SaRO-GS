# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ========================================================================================================
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the thirdparty/gaussian_splatting/LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys 
sys.path.append("./thirdparty/gaussian_splatting")

import torch
from thirdparty.gaussian_splatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
import time 
import scipy
import numpy as np 
import warnings
import json 
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image

from thirdparty.gaussian_splatting.lpipsPyTorch import lpips

from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.helper3dg import gettestparse
from skimage.metrics import structural_similarity as sk_ssim
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams
from thirdparty.gaussian_splatting.renderer import test_render
from thirdparty.gaussian_splatting.scene.saro_gaussian import GaussianModel
warnings.filterwarnings("ignore")

# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/render.py and https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py
def render_set(model_path, name, iteration, views, gaussians, background,require_segment,args):

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    if name != "val":
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
        segment_path = os.path.join(model_path, name, "ours_{}".format(iteration), "segment")

        makedirs(gts_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)
        makedirs(segment_path, exist_ok=True)

    statsdict = {}

    scales = gaussians.get_scaling

    scalemax = torch.amax(scales).item()
    scalesmean = torch.amin(scales).item()
     
    op = gaussians.get_opacity
    opmax = torch.amax(op).item()
    opmean = torch.mean(op).item()

    statsdict["scales_max"] = scalemax
    statsdict["scales_mean"] = scalesmean

    statsdict["op_max"] = opmax
    statsdict["op_mean"] = opmean 


    statspath = os.path.join(model_path, "stat_" + str(iteration) + ".json")
    with open(statspath, 'w') as fp:
            json.dump(statsdict, fp, indent=True)


    psnrs = []
    lpipss = []
    lpipssvggs = []

    full_dict = {}
    per_view_dict = {}
    ssims = []
    ssimsv2 = []
    scene_dir = model_path
    image_names = []
    times = []

    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}

  

    full_dict[scene_dir][iteration] = {}
    per_view_dict[scene_dir][iteration] = {}
 

    for idx, view in enumerate(tqdm(views, desc="Rendering and metric progress")):

        renderingpkg = test_render(view, gaussians, background,require_segment=require_segment) # C x H x W
        rendering = renderingpkg["render"]
        rendering = torch.clamp(rendering, 0, 1.0)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


        if name != "val":

            #save depth
            if "depth" in renderingpkg:
                depth_np = renderingpkg["depth"].squeeze().detach().cpu().numpy()
                plt.imsave(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), depth_np, cmap='viridis')
            #save segment
            if "segment_render" in renderingpkg:
                segment = renderingpkg["segment_render"]
                segment = torch.clamp(segment, 0, 1.0)
                torchvision.utils.save_image(segment, os.path.join(segment_path, '{0:05d}'.format(idx) + ".png"))

            gt = view.original_image[0:3, :, :].cuda().float()
            ssims.append(ssim(rendering,gt)) 
            psnrs.append(psnr(rendering, gt).mean().double().item())

            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        image_names.append('{0:05d}'.format(idx) + ".png")

    if name == "val":
        return
    for idx, view in enumerate(tqdm(views, desc="release gt images cuda memory for timing")):
        view.original_image = None #.detach()  
        torch.cuda.empty_cache()

    # start timing
    for _ in range(4):
        for idx, view in enumerate(tqdm(views, desc="timing ")):

            renderpack = test_render(view, gaussians, background)#["time"] # C x H x W
            duration = renderpack["duration"]
            if idx > 10: #warm up
                times.append(duration)

    print(np.mean(np.array(times)))
    if len(views) > 0:
        full_dict[model_path][iteration].update({"SSIM": torch.tensor(ssims).mean().item(),
                                        "PSNR": torch.tensor(psnrs).mean().item(),

                                        "times": torch.tensor(times).mean().item()})
        
        per_view_dict[model_path][iteration].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                       })
        
            
            
        with open(model_path + "/" + str(iteration) + "_runtimeresults.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)

        with open(model_path + "/" + str(iteration) + "_runtimeperview.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)



def run_test(dataset : ModelParams, ckpt, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_val: bool,require_segment : bool, duration: int,args, loader="colmap"):
    
    with torch.no_grad():
        print("use model {}".format(dataset.model))

        gaussians = GaussianModel(dataset)
        gaussians.duration = args.duration
        gaussians.preprocesspoints = args.preprocesspoints 

        if dataset.color_order >0:
            gaussians.color_order = dataset.color_order
            
        scene = Scene(dataset, gaussians, shuffle=False, multiview=False, duration=duration, loader=loader)

        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda") # use black background

        # (model_params, iteration) = torch.load(ckpt)
        # print("load at",iteration)
        gaussians.load_ply(ckpt)
        # gaussians.restore(model_params, None)
        # gaussians.save_ply("./flame_steak/ckpt_best.ply")
        gaussians.get_deformfeature()
        iteration = "best"
        if not skip_test:            
            render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, background, require_segment = require_segment, args=args)
        
        if not skip_val:            
            render_set(dataset.model_path, "val", iteration, scene.getValCameras(), gaussians, background,require_segment=False, args=args)
if __name__ == "__main__":
    

    args, model_extract, pp_extract =gettestparse()
    run_test(model_extract, args.checkpoint, pp_extract, args.skip_train, args.skip_test,args.skip_val,  args.require_segment, args.duration,  args,loader=args.valloader)