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

def getrenderpip(option="train_ours_full"):
    print("render option", option)
    if option == "train_ours_full":
        from thirdparty.gaussian_splatting.renderer import train_ours_full 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizer  
        return train_ours_full, GaussianRasterizationSettings, GaussianRasterizer


    elif option == "train_ours_lite":
        from thirdparty.gaussian_splatting.renderer import train_ours_lite
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer  

        return train_ours_lite, GaussianRasterizationSettings, GaussianRasterizer
    
    elif option == "test_ours_full":
        from thirdparty.gaussian_splatting.renderer import test_ours_full
        from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizer  
        return test_ours_full, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_ours_lite": # forward only 
        from thirdparty.gaussian_splatting.renderer import test_ours_lite
        from forward_lite import GaussianRasterizationSettings 
        from forward_lite import GaussianRasterizer 
        return test_ours_lite, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "train_ours_flow":
        from thirdparty.gaussian_splatting.renderer import train_ours_flow
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return train_ours_flow, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "test_ours_flow":
        from thirdparty.gaussian_splatting.renderer import test_ours_flow
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings #这里为什么是forward lite？
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return test_ours_flow, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "train_ours_flow_full":
        from thirdparty.gaussian_splatting.renderer import train_ours_flow_full
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return train_ours_flow_full, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "test_ours_flow_full":
        from thirdparty.gaussian_splatting.renderer import test_ours_flow_full
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings  
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer
        return test_ours_flow_full, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "train_ours_flow_mlp":
        from thirdparty.gaussian_splatting.renderer import train_ours_flow_mlp
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return train_ours_flow_mlp, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "test_ours_flow_mlp":
        from thirdparty.gaussian_splatting.renderer import test_ours_flow_mlp
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return test_ours_flow_mlp, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "train_ours_xyz_mlp":
        from thirdparty.gaussian_splatting.renderer import train_ours_xyz_mlp
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return train_ours_xyz_mlp, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "test_ours_xyz_mlp":
        from thirdparty.gaussian_splatting.renderer import test_ours_xyz_mlp
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return test_ours_xyz_mlp, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "train_ours_flow_mlp_opacity":
        from thirdparty.gaussian_splatting.renderer import train_ours_flow_mlp_opacity
        from diff_gaussian_rasterization import GaussianRasterizationSettings 
        from diff_gaussian_rasterization import GaussianRasterizer 
        partial_render = partial(train_ours_flow_mlp_opacity,GRzer=GaussianRasterizer,GRsetting=GaussianRasterizationSettings)
        return partial_render
    elif  option == "test_ours_flow_mlp_opacity":
        from thirdparty.gaussian_splatting.renderer import test_ours_flow_mlp_opacity
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return test_ours_flow_mlp_opacity, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "train_ours_bi_flow":
        from thirdparty.gaussian_splatting.renderer import train_ours_bi_flow
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return train_ours_bi_flow, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "test_ours_bi_flow":
        from thirdparty.gaussian_splatting.renderer import test_ours_bi_flow
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return test_ours_bi_flow, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "train_ours_bi_flow_alldeform":
        from thirdparty.gaussian_splatting.renderer import train_ours_bi_flow_alldeform
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return train_ours_bi_flow_alldeform, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "test_ours_bi_flow_alldeform":
        from thirdparty.gaussian_splatting.renderer import test_ours_bi_flow_alldeform
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer 
        return test_ours_bi_flow_alldeform, GaussianRasterizationSettings, GaussianRasterizer
    elif  option == "train_ours_flow_mlp_opacity_color":
        from thirdparty.gaussian_splatting.renderer import train_ours_flow_mlp_opacity_color 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizer
        partial_render = partial(train_ours_flow_mlp_opacity_color,GRzer=GaussianRasterizer,GRsetting=GaussianRasterizationSettings)
        return partial_render
    else:
        raise NotImplementedError("Rennder {} not implemented".format(option))
    
def getmodel(model="oursfull"):
    if model == "ours_full":
        from  thirdparty.gaussian_splatting.scene.oursfull import GaussianModel
    elif model == "ours_lite":
        from  thirdparty.gaussian_splatting.scene.ourslite import GaussianModel
    elif model =="flow":
        from  thirdparty.gaussian_splatting.scene.flow import GaussianModel
    elif model =="flow_static":
        from thirdparty.gaussian_splatting.scene.flow_staticmask import GaussianModel
    elif model =="flow_full":
        from thirdparty.gaussian_splatting.scene.flow_full import GaussianModel
    elif model =="flow_mlp":
        from thirdparty.gaussian_splatting.scene.flow_mlp import GaussianModel
    elif model =="xyz_mlp":
        from thirdparty.gaussian_splatting.scene.xyz_mlp import GaussianModel
    elif model =="xyz_mlp_enc":
        from thirdparty.gaussian_splatting.scene.xyz_mlp_enc import GaussianModel
    elif model=="flow_mlp_hexplane":
        from thirdparty.gaussian_splatting.scene.flow_mlp_hexplane import GaussianModel
    elif model=="flow_mlp_hexplane_motion":
        from thirdparty.gaussian_splatting.scene.flow_mlp_hexplane_motion import GaussianModel
    elif model=="flow_mlp_hexplane_opacity":
        from thirdparty.gaussian_splatting.scene.flow_mlp_hexplane_opacity import GaussianModel
    elif model=="flow_mlp_hexplane_basemlp":
        from thirdparty.gaussian_splatting.scene.flow_mlp_hexplane_basemlp import GaussianModel
    elif model=="bi_flow":
        from thirdparty.gaussian_splatting.scene.bi_flow import GaussianModel
    elif model=="bi_flow_alldeform":
        from thirdparty.gaussian_splatting.scene.bi_flow_alldeform import GaussianModel
    else:
        try:
            model_name =  f"thirdparty.gaussian_splatting.scene.{model}"
            imported_module = importlib.import_module(model_name)
            GaussianModel  = getattr(imported_module,"GaussianModel")
            # print(GaussianModel)
        except Exception as e:
            print(e)
            raise NotImplementedError("model {} not implemented".format(model))
    return GaussianModel

def getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii,timestamp,iteration,lambda_all):
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
    # if opt.reg == 1: # add optical flow loss
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._motion) / gaussians._motion.shape[0]
    # elif opt.reg == 0 :
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))   
    # elif opt.reg == 9 : #regulizor on the rotation
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._omega[radii>0]**2)
    # elif opt.reg == 10 : #regulizor on the rotation
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._motion[radii>0]**2)
    # elif opt.reg == 4:
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians.get_scaling) / gaussians._motion.shape[0]
    # elif opt.reg == 5:
    #     loss = Ll1  
    # elif opt.reg == 6 :
    #     ratio = torch.mean(gt_image) - 0.5 + opt.lambda_dssim
    #     ratio = torch.clamp(ratio, 0.0, 1.0)
    #     loss = (1.0 - ratio) * Ll1 + ratio * (1.0 - ssim(image, gt_image))
    # elif opt.reg == 7 :
    #     Ll1 = Ll1 / (torch.mean(gt_image) * 2.0) # normalize L1 loss
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    # elif opt.reg == 8 :
    #     N = gaussians._xyz.shape[0]
    #     mean = torch.mean(gaussians._xyz, dim=0, keepdim=True)
    #     varaince = (mean - gaussians._xyz)**2 #/ N
    #     loss = (1.0 - opt.lambda_dssim) * Ll1  + 0.0002*opt.lambda_dssim * torch.sum(varaince) / N
    # elif opt.reg == 11:
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * gaussians.get_motion_l1loss() +opt.regl*gaussians.get_rot_l1loss()+ opt.regl * gaussians.get_time_smoothloss(timestamp)
    # elif opt.reg ==12:
    #     print("reg 12")
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * gaussians.get_time_smoothloss(timestamp)
    # elif opt.reg ==13:
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * gaussians.get_motion_l1loss() +opt.regl*gaussians.get_rot_l1loss()
    # elif opt.reg ==14:
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * gaussians.get_time_smoothloss(timestamp)
    #     if iteration >= 20000:
    #         loss = loss + opt.regl * gaussians.get_rigid_loss()
    # elif opt.reg==15:
    #     '''加入正则优化dynamatic'''
    #     dynamatic_entropy = -(gaussians.dynamatic * torch.log(gaussians.dynamatic+1e-36) + (1-gaussians.dynamatic)*torch.log((1 -gaussians.dynamatic + 1e-36)))
    #     # print(dynamatic_entropy)
    #     dynamatic_entropy=dynamatic_entropy.mean(dim=0)
    #     # print(dynamatic_entropy)
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * dynamatic_entropy
    # elif opt.reg==16:
    #     dynamatic_probs = torch.nn.functional.softmax(gaussians.dynamatic,dim=0).squeeze()
    #     print(dynamatic_probs,dynamatic_probs.amax())
    #     categorical_dist = torch.distributions.Categorical(dynamatic_probs)
    #     # print(categorical_dist)
    #     dynamatic_entropy = categorical_dist.entropy()
    #     # print(dynamatic_entropy.item())
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * dynamatic_entropy
    # return loss 


def freezweights(model, screenlist):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad = torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  

def freezweightsbymask(model, screenlist, mask):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad =  mask.unsqueeze(1)*grad_tensor #torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  


def freezweightsbymasknounsqueeze(model, screenlist, mask):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad =  mask*grad_tensor #torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  





def controlgaussians(opt, gaussians, densify, iteration, scene,  visibility_filter, radii, viewspace_point_tensor, flag, traincamerawithdistance=None, maxbounds=None, minbounds=None): 
    if densify == 1: # n3d 
        if iteration < opt.densify_until_iter :
            if iteration ==  8001 : # 8001
                omegamask = gaussians.zero_omegabymotion() # 1 we keep omega, 0 we freeze omega
                gaussians.omegamask  = omegamask
                scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            elif iteration > 8001: # 8001
                freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask) #将_omega中被mask掉的梯度设为0
                rotationmask = torch.logical_not(gaussians.omegamask) #将没有被omega mask掉的旋转部分的梯度设为0
                freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.desicnt: #只clone这么多次
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    #clone and pure
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    if iteration < 7000 : # defalt 7000. 
                        prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                        scene.recordpoints(iteration, "addionally prune_mask")
            if iteration % 3000 == 0 :
                gaussians.reset_opacity()
        else:
            freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
            rotationmask = torch.logical_not(gaussians.omegamask)
            freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask) #uncomment freezeweight... for fast traning speed.
            if iteration % 1000 == 500 :
                zmask = gaussians._xyz[:,2] < 4.5  # 
                gaussians.prune_points(zmask) 
                torch.cuda.empty_cache()
            if iteration == 10000: 
                removeminmax(gaussians, maxbounds, minbounds)
        return flag
    
    elif densify == 2: # n3d 
        if iteration < opt.densify_until_iter :
            #对旋转施加mask，暂时不知道有什么用
            # if iteration ==  8001 : # 8001
            #     omegamask = gaussians.zero_omegabymotion() #
            #     gaussians.omegamask  = omegamask
            #     scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            # elif iteration > 8001: # 8001
            #     freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
            #     rotationmask = torch.logical_not(gaussians.omegamask)
            #     freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)


            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if (opt.desicnt < 0 or flag < opt.desicnt )and (opt.max_points_num<0 or gaussians.get_points_num < opt.max_points_num): #最多的densify次数,小于0表示这个参数没用.max_points_num表示最多的点数,小于-1表示参数没用
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # size_threshold = 20
                    # z_mask = (gaussians.get_xyz[:,2] < 4.5).squeeze()
                    # print("pure_z",z_mask.sum())
                    # gaussians.prune_points(z_mask)

                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                    
                    if hasattr(gaussians,"valid_mask") and gaussians.valid_mask is not None:
                        valid_mask = ~prune_mask
                        gaussians.valid_mask = torch.logical_and(valid_mask,gaussians.valid_mask)

                        #将左右两边为false的点去掉
                        # right_shift=torch.cat((torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool),self.valid_mask[:,:-1]),dim=1)
                        # left_shift=torch.cat((self.valid_mask[:,1:],torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool)),dim=1)
                        # self.valid_mask = torch.logical_and(self.valid_mask,torch.logical_or(right_shift,left_shift)) #左右两边有一个为true，就将这个点保留
                        prune_mask = torch.all(~gaussians.valid_mask,dim=1) #如果全为false，则为true #[N]
                    
                    gaussians.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                    scene.recordpoints(iteration, "addionally prune_mask")
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
        return flag
    

    elif densify == 3: # techni
        if iteration < opt.densify_until_iter :
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.desicnt:
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    if iteration < 7000 : # defalt 7000. 
                        prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                        scene.recordpoints(iteration, "addionally prune_mask")
            if iteration % opt.opacity_reset_interval == 0 :
                gaussians.reset_opacity()
        else:
            if iteration == 10000: 
                removeminmax(gaussians, maxbounds, minbounds)
        return flag

    elif densify == 4: # n3d 
        if iteration < opt.densify_until_iter :
            #对旋转施加mask，暂时不知道有什么用
            # if iteration ==  8001 : # 8001
            #     omegamask = gaussians.zero_omegabymotion() #
            #     gaussians.omegamask  = omegamask
            #     scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            # elif iteration > 8001: # 8001
            #     freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
            #     rotationmask = torch.logical_not(gaussians.omegamask)
            #     freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)


            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if (opt.desicnt < 0 or flag < opt.desicnt )and (opt.max_points_num<0 or gaussians.get_points_num < opt.max_points_num): #最多的densify次数,小于0表示这个参数没用.max_points_num表示最多的点数,小于-1表示参数没用
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                    
                    if hasattr(gaussians,"valid_mask") and gaussians.valid_mask is not None:
                        valid_mask = ~prune_mask
                        gaussians.valid_mask = torch.logical_and(valid_mask,gaussians.valid_mask)

                        #将左右两边为false的点去掉
                        # right_shift=torch.cat((torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool),self.valid_mask[:,:-1]),dim=1)
                        # left_shift=torch.cat((self.valid_mask[:,1:],torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool)),dim=1)
                        # self.valid_mask = torch.logical_and(self.valid_mask,torch.logical_or(right_shift,left_shift)) #左右两边有一个为true，就将这个点保留
                        prune_mask = torch.all(~gaussians.valid_mask,dim=1) #如果全为false，则为true #[N]
                    
                    gaussians.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                    scene.recordpoints(iteration, "addionally prune_mask")
            if iteration == opt.coarse_opacity_reset_interval:
                print("coarse reset opacity")
                gaussians.reset_opacity()

            if (iteration-opt.coarse_opacity_reset_interval) % opt.opacity_reset_interval == 0 :
                print("reset opacity")
                gaussians.reset_opacity()
        if iteration == opt.pure_static:
                print("pure static")
                gaussians.pure_static_points()
                # print("enable time downsample")
                # gaussians.hexplane.convert_to_timedonwsample()
                # gaussians.init_mlp_grd()
                # gaussians.hexplane.enable_time_downsample = True
        # else:
        #     if iteration % 1000 == 500 :
        #         zmask = gaussians._xyz[:,2] < 4.5  # for stability  
        #         gaussians.prune_points(zmask) 
        #         torch.cuda.empty_cache()
        return flag
    elif densify == 5: # dnerf 
        if iteration < opt.densify_until_iter :

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if (opt.desicnt < 0 or flag < opt.desicnt )and (opt.max_points_num<0 or gaussians.get_points_num < opt.max_points_num): #最多的densify次数,小于0表示这个参数没用.max_points_num表示最多的点数,小于-1表示参数没用
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                     
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                    
                    if hasattr(gaussians,"valid_mask") and gaussians.valid_mask is not None:
                        valid_mask = ~prune_mask
                        gaussians.valid_mask = torch.logical_and(valid_mask,gaussians.valid_mask)

                        #将左右两边为false的点去掉
                        # right_shift=torch.cat((torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool),self.valid_mask[:,:-1]),dim=1)
                        # left_shift=torch.cat((self.valid_mask[:,1:],torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool)),dim=1)
                        # self.valid_mask = torch.logical_and(self.valid_mask,torch.logical_or(right_shift,left_shift)) #左右两边有一个为true，就将这个点保留
                        prune_mask = torch.all(~gaussians.valid_mask,dim=1) #如果全为false，则为true #[N]
                    
                    gaussians.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                    scene.recordpoints(iteration, "addionally prune_mask")
            # print( opt.opacity_reset_interval+1)
            if iteration % (opt.opacity_reset_interval) == 0 :
                print("reset opacity")
                gaussians.reset_opacity()

        return flag
    elif densify == 6: # hyper
        if iteration < opt.densify_until_iter :
            #对旋转施加mask，暂时不知道有什么用
            # if iteration ==  8001 : # 8001
            #     omegamask = gaussians.zero_omegabymotion() #
            #     gaussians.omegamask  = omegamask
            #     scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            # elif iteration > 8001: # 8001
            #     freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
            #     rotationmask = torch.logical_not(gaussians.omegamask)
            #     freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)


            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if (opt.desicnt < 0 or flag < opt.desicnt )and (opt.max_points_num<0 or gaussians.get_points_num < opt.max_points_num): #最多的densify次数,小于0表示这个参数没用.max_points_num表示最多的点数,小于-1表示参数没用
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # size_threshold = 20
                    # z_mask = (gaussians.get_xyz[:,2] < 4.5).squeeze()
                    # print("pure_z",z_mask.sum())
                    # gaussians.prune_points(z_mask)

                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                    
                    if hasattr(gaussians,"valid_mask") and gaussians.valid_mask is not None:
                        valid_mask = ~prune_mask
                        gaussians.valid_mask = torch.logical_and(valid_mask,gaussians.valid_mask)

                        #将左右两边为false的点去掉
                        # right_shift=torch.cat((torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool),self.valid_mask[:,:-1]),dim=1)
                        # left_shift=torch.cat((self.valid_mask[:,1:],torch.zeros((self.valid_mask.shape[0],1),device="cuda",dtype=bool)),dim=1)
                        # self.valid_mask = torch.logical_and(self.valid_mask,torch.logical_or(right_shift,left_shift)) #左右两边有一个为true，就将这个点保留
                        prune_mask = torch.all(~gaussians.valid_mask,dim=1) #如果全为false，则为true #[N]
                    
                    gaussians.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                    scene.recordpoints(iteration, "addionally prune_mask")
            # print( opt.opacity_reset_interval+1)
            if iteration % (opt.opacity_reset_interval) == 0 :
                print("reset opacity")
                gaussians.reset_opacity()

        # else:
        #     if iteration % 500 == 1 :
        #         zmask = gaussians.real_xyz[:,2] < 4.5  # for stability  
        #         print("pure realxyz：",torch.sum(zmask).item())
        #         gaussians.prune_points(zmask) 
        #         torch.cuda.empty_cache()
        return flag
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




def reloadhelper(gaussians, opt, maxx, maxy, maxz,  minx, miny, minz):
    givenpath = opt.prevpath
    if opt.loadall == 0:
        gaussians.load_plyandminmax(givenpath, maxx, maxy, maxz,  minx, miny, minz)
    elif opt.loadall == 1 :
        gaussians.load_plyandminmaxall(givenpath, maxx, maxy, maxz,  minx, miny, minz)
    elif opt.loadall == 2 :        
        gaussians.load_ply(givenpath)
    elif opt.loadall == 3:
        gaussians.load_plyandminmaxY(givenpath, maxx, maxy, maxz,  minx, miny, minz)

    gaussians.max_radii2D =  torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    return 

def getfisheyemapper(folder, cameraname):
    parentfolder = os.path.dirname(folder)
    distoritonflowpath = os.path.join(parentfolder, cameraname + ".npy")
    distoritonflow = np.load(distoritonflowpath)
    distoritonflow = torch.from_numpy(distoritonflow).unsqueeze(0).float().cuda()
    return distoritonflow











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
