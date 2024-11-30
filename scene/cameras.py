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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV
from utils.graphics_utils import fov2focal
# def fov2focal(fov, pixels):
#     return pixels / (2 * math.tan(fov / 2))
from kornia import create_meshgrid
from helper_model import pix2ndc
import random 
import sys

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", near=0.01, far=100.0, timestamp=0.0, rayo=None, rayd=None, rays=None, cxr=0.0,cyr=0.0,mask=None,meta_only=False,
                 resolution=None,image_path=None
                 ):
        super(Camera, self).__init__()
        # print("build camera")
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        # print(image.device)


        self.image_name = image_name
        self.image_path = image_path
        self.timestamp = timestamp
        self.meta_only =meta_only #是meta only，就表示目前image是PIL格式。不是就表示变成了tensor
        self.resolution = resolution #如果是meta_only,则表示是没有剪裁过的，要在dataset里面再剪裁
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        #4dgs
        # print(meta_only)
        if meta_only:
            # print(1)
            # self.original_image = np.array(image) #这个时候是PIL image,但是要转成numpy.因为用pil读取会有问题，但是太慢
            self.image_width = resolution[0] #这里记录剪裁过的分辨率
            self.image_height = resolution[1]
        else:
            #spt code, image is real image 
            if not isinstance(image, tuple):
                if "camera_" not in image_name:
                    self.original_image = image.clamp(0.0, 1.0)
                    # .to(self.data_device)
                else:
                    self.original_image = image.clamp(0.0, 1.0).half()
                    # .to(self.data_device)
                self.image_width = self.original_image.shape[2]
                self.image_height = self.original_image.shape[1]
                if gt_alpha_mask is not None:
                    self.original_image *= gt_alpha_mask
                    # .to(self.data_device)
                else:
                    self.original_image *= torch.ones((1, self.image_height, self.image_width))
                    # , device=self.data_device)
            #not real
            else:
                self.image_width = image[0]
                self.image_height = image[1]
                self.original_image = None
        
        # print(sys.getsizeof(self.original_image)) n3d一张图片要72字节
        
        #使用open3d算轨迹需要用到
        self.focalx = fov2focal(FoVx, self.image_width)
        self.focaly = fov2focal(FoVy, self.image_height)
        
        self.mask = mask
        self.zfar = 100.0
        self.znear = 0.01  

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        #.cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1)
            #.cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
            #.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0) #这个含义是，先将世界坐标系的点变换到相机坐标系中，再将其变换到NDC坐标空间中。
        #(A.T@B.T)=(B@A).T
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # if not meta_only:
        #     self.projection_matrix = self.projection_matrix.cuda()
        #     self.world_view_transform = self.world_view_transform.cuda()
        #     self.full_proj_transform = self.full_proj_transform.cuda()
        #     self.camera_center = self.world_view_transform.inverse()[3, :3].cuda()

        self.rayo = None
        self.rayd = None
        # if rayd is not None:
        #     projectinverse = self.projection_matrix.T.inverse()
        #     camera2wold = self.world_view_transform.T.inverse()
        #     pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
        #     pixgrid = pixgrid
        #     #.cuda()  # H,W,
            
        #     xindx = pixgrid[:,:,0] # x 
        #     yindx = pixgrid[:,:,1] # y
      
            
        #     ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
        #     ndcx = ndcx.unsqueeze(-1)
        #     ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
        #     ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

        #     projected = ndccamera @ projectinverse.T #将NDC坐标系下的点变换到相机坐标系下
        #     diretioninlocal = projected / projected[:,:,3:] #v 


        #     direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T #将相机坐标系下的点变换到世界坐标系下
        #     rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            
        #     self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
        #     self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)    
        #     self.rays = torch.cat([self.rayo, self.rayd], dim=1)
        #     #.cuda()
            

        # else :
        #     self.rayo = None
        #     self.rayd = None
    def __copy__(self):
        # 创建一个新对象，传递当前对象的属性给它
        new_obj = type(self).__new__(self.__class__)
        new_obj.__dict__.update(self.__dict__)
        return new_obj



class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class Camerass(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", near=0.01, far=100.0, timestamp=0.0, rayo=None, rayd=None, rays=None, cxr=0.0,cyr=0.0,
                 ):
        super(Camerass, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        self.fisheyemapper = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # image is real image 
        if not isinstance(image, tuple):
            if "camera_" not in image_name:
                self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            else:
                self.original_image = image.clamp(0.0, 1.0).half().to(self.data_device)
            print("read one")# lazy loader?
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

        else:
            self.image_width = image[0] 
            self.image_height = image[1] 
            self.original_image = None
        
        self.image_width = 2 * self.image_width
        self.image_height = 2 * self.image_height # 

        self.zfar = 100.0
        self.znear = 0.01  
        self.trans = trans
        self.scale = scale

        # w2c 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
      
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] # 

            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)                                                                          #rayd.permute(2, 0, 1).unsqueeze(0)
        else :
            self.rayo = None
            self.rayd = None
