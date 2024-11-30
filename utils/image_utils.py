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

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    # print("fafaf",img1.shape)
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    # print(mse.min(), mse.max(),mse.shape)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnrmask(img1, img2):
    #img1 = img1.squeeze(0)
    #img2 = img2.squeeze(0)
    mask = img2 > 0.0
    validmask = torch.sum(img2[:, :, :], dim=1) > 0.01
    validmask = validmask.repeat(3, 1, 1)#.float()
    validmask = validmask.view(img1.shape[0], -1)
    
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1)[:, validmask[0,:]].mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
# def psnrmask(img, img2, mask):
#     mse = (((img - mask)) ** 2).view(img.shape[0], -1).mean(1, keepdim=True)
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))

def easy_cmap(x: torch.Tensor):
    x_rgb = torch.zeros((3, x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
    x_max, x_min = x.max(), x.min()
    x_normalize = (x - x_min) / (x_max - x_min)
    x_rgb[0] = torch.clamp(x_normalize, 0, 1)
    x_rgb[1] = torch.clamp(x_normalize, 0, 1)
    x_rgb[2] = torch.clamp(x_normalize, 0, 1)
    return x_rgb