#
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

# =============================================

# This license is additionally subject to the following restrictions:

# Licensor grants non-exclusive rights to use the Software for research purposes
# to research users (both academic and industrial), free of charge, without right
# to sublicense. The Software may be used "non-commercially", i.e., for research
# and/or evaluation purposes only.

# Subject to the terms and conditions of this License, you are granted a
# non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
# publicly display, publicly perform and distribute its Work and any resulting
# derivative works in any form.
#

import torch
import numpy as np
import torch
from simple_knn._C import distCUDA2
import os 
import json 
import cv2
# from script.pre_immersive_distorted import SCALEDICT 
from functools import partial
import importlib

def getloss(opt, Ll1, ssim, image, gt_image, gaussians,lambda_all):
    if opt.lambda_dssim >0:
        Ldssim = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Ldssim
    else:
        loss = Ll1
    if opt.lambda_dtstd >0:
        Ldtstd = 1-gaussians.get_dynamatic_trbfcenter.std()
        loss = loss + opt.lambda_dtstd * Ldtstd
    if opt.lambda_dl1_opacity>0:
        Ldl1_opacity = gaussians.get_trbfscale.mean()
        loss = loss + opt.lambda_dl1_opacity * Ldl1_opacity
    if opt.lambda_dscale_entropy>0:
        scale_entropy = -(gaussians.get_trbfscale * torch.log(gaussians.get_trbfscale+1e-36) + (1-gaussians.get_trbfscale)*torch.log((1 -gaussians.get_trbfscale + 1e-36)))
        Ldscale_entropy=scale_entropy.mean(dim=0)
        loss = loss + opt.lambda_dscale_entropy * Ldscale_entropy
    # print(opt.lambda_dscale_reg)
    if opt.lambda_dscale_reg>0:
        if gaussians.is_dynamatic and gaussians.scale_residual != None:
            Ldscale_reg = torch.linalg.vector_norm(gaussians.scale_residual , ord=2)
            loss = loss + opt.lambda_dscale_reg * Ldscale_reg
        else:
            Ldscale_reg = torch.tensor([0])

    if opt.lambda_dshs_reg>0:
        # print(gaussians.active_sh_degree)
        Ldshs_reg = torch.linalg.matrix_norm(gaussians.shs_residual[:,:(gaussians.active_sh_degree+1)**2].reshape(gaussians._xyz.shape[0],-1) )
        # print(Ldshs_reg)
        loss = loss + opt.lambda_dshs_reg * Ldshs_reg
    if opt.lambda_dmotion_reg>0:
        Ldmotion_reg = torch.linalg.matrix_norm(gaussians.motion_residual)
        loss = loss + opt.lambda_dmotion_reg * Ldmotion_reg

    if opt.lambda_dplanetv>0:
        Ldplanetv = gaussians.hexplane.planetv()
        loss += opt.lambda_dplanetv * Ldplanetv
    if opt.lambda_dtime_smooth>0:
        Ldtime_smooth = gaussians.hexplane.timesmooth()
        loss += opt.lambda_dtime_smooth*Ldtime_smooth

    #记录各种loss
    loss_dict ={"Ll1":Ll1}
    with torch.no_grad():
        for lambda_name in lambda_all:
            if opt.__dict__[lambda_name] > 0:
                # ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                # vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * vars()[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema
                loss_dict[lambda_name.replace("lambda_", "L")] = vars()[lambda_name.replace("lambda_", "L")]
        # print(loss_dict)
        return loss, loss_dict



def controlgaussians(opt, gaussians, densify, iteration, scene): 

    
    if densify == 2: # n3d 
        if iteration < opt.densify_until_iter :

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                # if (opt.desicnt < 0 or flag < opt.desicnt )and (opt.max_points_num<0 or gaussians.get_points_num < opt.max_points_num): #最多的densify次数,小于0表示这个参数没用.max_points_num表示最多的点数,小于-1表示参数没用
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    scene.recordpoints(iteration, "after densify")
                # else:
                #     prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                    
                #     if hasattr(gaussians,"valid_mask") and gaussians.valid_mask is not None:
                #         valid_mask = ~prune_mask
                #         gaussians.valid_mask = torch.logical_and(valid_mask,gaussians.valid_mask)

                #         #将左右两边为false的点去掉
                #         # right_shift=torch.cat((torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool),self.valid_mask[:,:-1]),dim=1)
                #         # left_shift=torch.cat((self.valid_mask[:,1:],torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool)),dim=1)
                #         # self.valid_mask = torch.logical_and(self.valid_mask,torch.logical_or(right_shift,left_shift)) #左右两边有一个为true，就将这个点保留
                #         prune_mask = torch.all(~gaussians.valid_mask,dim=1) #如果全为false，则为true #[N]
                    
                #     gaussians.prune_points(prune_mask)
                #     torch.cuda.empty_cache()
                #     scene.recordpoints(iteration, "addionally prune_mask")
            # print( opt.opacity_reset_interval+1)
            if iteration % (opt.opacity_reset_interval) == 0 :
                print("reset opacity")
                gaussians.reset_opacity()

        else:
            if iteration % 500 == 1 :
                zmask = gaussians.real_xyz[:,2] < 4.5  # for stability  
                print("pure realxyz：",torch.sum(zmask).item())
                gaussians.prune_points(zmask) 
                torch.cuda.empty_cache()
    
    elif densify == 5: # dnerf 
        if iteration < opt.densify_until_iter :

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                # if (opt.desicnt < 0 or flag < opt.desicnt )and (opt.max_points_num<0 or gaussians.get_points_num < opt.max_points_num): #最多的densify次数,小于0表示这个参数没用.max_points_num表示最多的点数,小于-1表示参数没用
                scene.recordpoints(iteration, "before densify")
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    
                gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                scene.recordpoints(iteration, "after densify")
                # else:
                #     prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                    
                #     if hasattr(gaussians,"valid_mask") and gaussians.valid_mask is not None:
                #         valid_mask = ~prune_mask
                #         gaussians.valid_mask = torch.logical_and(valid_mask,gaussians.valid_mask)

                #         #将左右两边为false的点去掉
                #         # right_shift=torch.cat((torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool),self.valid_mask[:,:-1]),dim=1)
                #         # left_shift=torch.cat((self.valid_mask[:,1:],torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool)),dim=1)
                #         # self.valid_mask = torch.logical_and(self.valid_mask,torch.logical_or(right_shift,left_shift)) #左右两边有一个为true，就将这个点保留
                #         prune_mask = torch.all(~gaussians.valid_mask,dim=1) #如果全为false，则为true #[N]
                    
                #     gaussians.prune_points(prune_mask)
                #     torch.cuda.empty_cache()
                #     scene.recordpoints(iteration, "addionally prune_mask")
            # print( opt.opacity_reset_interval+1)
            if iteration % (opt.opacity_reset_interval) == 0 :
                print("reset opacity")
                gaussians.reset_opacity()


      
def logicalorlist(listoftensor):
    mask = None 
    for idx, ele in enumerate(listoftensor):
        if idx == 0 :
            mask = ele 
        else:
            mask = torch.logical_or(mask, ele)
    return mask 



def recordpointshelper(model_path, numpoints, iteration, string):
    txtpath = os.path.join(model_path, "exp_log.txt")
    
    with open(txtpath, 'a') as file:
        file.write("iteration at "+ str(iteration) + "\n")
        file.write(string + " pointsnumber " + str(numpoints) + "\n")




def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0












def undistortimage(imagename, datasetpath,data):
    


    video = os.path.dirname(datasetpath) # upper folder 
    with open(os.path.join(video + "/models.json"), "r") as f:
                meta = json.load(f)

    for idx , camera in enumerate(meta):
        folder = camera['name'] # camera_0001
        view = camera
        intrinsics = np.array([[view['focal_length'], 0.0, view['principal_point'][0]],
                            [0.0, view['focal_length'], view['principal_point'][1]],
                            [0.0, 0.0, 1.0]])
        dis_cef = np.zeros((4))

        dis_cef[:2] = np.array(view['radial_distortion'])[:2]
        if folder != imagename:
             continue
        print("done one camera")
        map1, map2 = None, None
        sequencename = os.path.basename(video)
        focalscale = SCALEDICT[sequencename]
 
        h, w = data.shape[:2]


        image_size = (w, h)
        knew = np.zeros((3, 3), dtype=np.float32)

def trbfunction(x): 
    #阶段指数函数
    return torch.exp(-1*x.pow(2))
