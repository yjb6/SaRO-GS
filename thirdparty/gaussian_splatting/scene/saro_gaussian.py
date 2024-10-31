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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from sklearn.neighbors import KDTree
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, update_quaternion
from helper_model import getcolormodel, interpolate_point, interpolate_partuse,interpolate_pointv3,add_extra_point,prune_point

from scene.hexplane_mip import HexPlaneField_mip
from scene.hexplane_mip_allscale import HexPlaneField_mip_allscale
# import matplotlib.pyplot as plt
# import tinycudann as tcnn

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def params_init(self):
        torch.nn.init.xavier_uniform_(self.motion_fc1.weight)
        def xavier_init(m):
            for name,W in m.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(W, 1)
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(W)
        xavier_init(self.motion_mlp)
        xavier_init(self.rot_mlp)



    def __init__(self, args):
        self.args=args
        # self.args.dynamatic_mlp= False
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._motion = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._omega = torch.empty(0) #旋转的系数

    
        self.setup_functions()

        self.preprocesspoints = False

        
        self.duration = 0.0


        self.D=args.deform_feature_dim #16
        self.H=args.deform_hidden_dim #128
        self.time_emb ,out_dim = get_embedder(args.deform_time_encode) #8
        self.dir_emb, dir_out_dim = get_embedder(4,3)

        print(self.args.planemodel)
        if self.args.planemodel == "hexplane_mip":
            self.hexplane = HexPlaneField_mip(args.bounds, args.kplanes_config, args.multires)

        # elif self.args.planemodel == "hexplane_mip_allscale":
        #     self.hexplane = HexPlaneField_mip_allscale(args.bounds, args.kplanes_config, args.multires)
        else:
            raise NotImplementedError
        hexplane_outdim = self.hexplane.feat_dim

        self.motion_mlp = nn.Sequential(nn.Linear(out_dim+hexplane_outdim,self.H),nn.ReLU(),nn.Linear(self.H,self.H),nn.ReLU(),nn.Linear(self.H, 3))
        self.opacity_mlp = nn.Sequential(nn.Linear(hexplane_outdim,self.H),nn.ReLU(),nn.Linear(self.H,int(self.H/2) ),nn.ReLU(),nn.Linear(int(self.H/2), 1),nn.Sigmoid())#考虑整合进某个别的mlp中


        self.rot_mlp = nn.Sequential(nn.Linear(out_dim+hexplane_outdim,self.H),nn.ReLU(),nn.Linear(self.H,self.H),nn.ReLU(),nn.Linear(self.H,7))
        
        self.shs_mlp = nn.Sequential(nn.Linear(out_dim+hexplane_outdim,self.H),nn.ReLU(),nn.Linear(self.H,self.H),nn.ReLU(),nn.Linear(self.H, 48))

        self._trbf_center =None
        self._trbf_scale =None


        self.min_intergral = self.args.min_intergral #最小的intergral，小于这个的应该被过滤掉
        self.is_dynamatic = False

    def capture(self):
        self.opt_dict = self.optimizer.state_dict()
        attributes = [
        "active_sh_degree",
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "max_radii2D",
        "xyz_gradient_accum",
        "denom",
        "opt_dict",
        "spatial_lr_scale",
        "motion_mlp",
        "rot_mlp",
        "opacity_mlp",
        "shs_mlp" if self.args.dsh else None,
        "scale_mlp" if self.args.dscale else None,
        "rgbdecoder" if self.args.rgbdecoder else None,
        "hexplane",
        "_motion",
        "_omega",
        "_trbf_center",
        "_trbf_scale"
    ]

    # 动态构建返回值元组
        return_values = tuple(
            getattr(self, attr_name)  for attr_name in attributes if attr_name)
        return return_values

    
    def restore(self, model_args, training_args):
        attributes = [
            ("active_sh_degree", self),
            ("_xyz", self),
            ("_features_dc", self),
            ("_features_rest", self),
            ("_scaling", self),
            ("_rotation", self),
            ("_opacity", self),
            ("max_radii2D", self),
            ("xyz_gradient_accum", None),  # 这个属性不是 self 的成员
            ("denom", None),               # 这个属性不是 self 的成员
            ("opt_dict", None),            # 这个属性不是 self 的成员
            ("spatial_lr_scale", self),
            ("motion_mlp", self),
            ("rot_mlp", self),
            ("opacity_mlp", self),
            ("shs_mlp" if self.args.dsh else None, self),
            ("scale_mlp" if self.args.dscale else None, self),
            ("rgbdecoder" if self.args.rgbdecoder else None,self),
            ("hexplane",self),
            ("_motion", self),
            ("_omega", self),
            ("_trbf_center", self),
            ("_trbf_scale", self)
        ]
        local_={}
        attributes = [attr for attr in attributes[:] if attr[0] is not None]
        for i, attr in enumerate(attributes):
            if attr[1] is not None:
                setattr(attr[1], attr[0], model_args[i])
            else:
                local_[attr[0]] = model_args[i]

        if self.is_dynamatic:
            with torch.no_grad():
                scales = torch.cat((self.get_scaling,self._trbf_scale.detach()/2),dim=1)
                hexplane_feature = self.hexplane(self._xyz,self._trbf_center ,scales) #[N,D]
                trbf_scale = 1-self.opacity_mlp(hexplane_feature) #得到trbf_scale
                min_scale = self.args.min_interval/(self.duration)
                trbf_scale = (1-min_scale)*trbf_scale + min_scale #限制min_scale最小值
                self._trbf_scale = trbf_scale
        print(self.hexplane.aabb)
        self.init_mlp_grd()
        if self.args.enable_scale_sum:
            self.init_real_scale()
        if training_args:#test的适合为None
            self.training_setup(training_args)
            self.xyz_gradient_accum = local_["xyz_gradient_accum"]
            
            self.denom =  local_["denom"]
            self.optimizer.load_state_dict(local_["opt_dict"])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    @property
    def get_rotation(self):
        #获取新的旋转
        return self.rotation_activation(self._rotation)
    


    @property
    def get_xyz(self):
        return self._xyz
    @property
    def get_trbfcenter(self):
        if self.args.sigmoid_tcenter:
            return torch.sigmoid(self._trbf_center)
        else:
            return self._trbf_center
    @property
    def get_dynamatic_trbfcenter(self):
        return self.get_trbfcenter
    @property
    def get_trbfscale(self):
        return self._trbf_scale
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, all_frame_time_init = False):
        '''要保证所有的参数都是叶子结点。如果有不是叶子节点的,那么在反传的时候就会报错。nn.parameter就能保证是个叶子节点'''
        print(self.preprocesspoints)
        if self.preprocesspoints == 0:
            pass
        elif self.preprocesspoints == 3:
            pcd = interpolate_point(pcd, 40) 
            pcd = add_extra_point(pcd,5000,100,0)
            pcd = prune_point(pcd,maxz=300)

        elif self.preprocesspoints == 31:
            pcd = interpolate_point(pcd, 40) 
            pcd = prune_point(pcd,maxz=200)

        elif self.preprocesspoints == 4:
            pcd = interpolate_point(pcd, 40) 
        
        elif self.preprocesspoints == 5:
            pcd = interpolate_point(pcd, 6) 

        elif self.preprocesspoints == 6:
            pcd = interpolate_point(pcd, 8) 
        
        elif self.preprocesspoints == 7:
            pcd = interpolate_point(pcd, 16) 
        
        elif self.preprocesspoints == 8:
            pcd = interpolate_pointv3(pcd, 4) 

        elif self.preprocesspoints == 14:
            pcd = interpolate_partuse(pcd, 2) 
        
        elif self.preprocesspoints == 15:
            pcd = interpolate_partuse(pcd, 4) 

        elif self.preprocesspoints == 16:
            pcd = interpolate_partuse(pcd, 8) 
        
        elif self.preprocesspoints == 17:
            pcd = interpolate_partuse(pcd, 16) 
        else:
            pcd = interpolate_point(pcd, self.preprocesspoints) 
        # else:
        #     pcd = pcd
        # pcd = add_extra_point(pcd)
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        # print(fused_point_cloud)
        # fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()

        #启用时间


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        if self.args.rgbdecoder:
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            features9channel = torch.cat((fused_color, fused_color, fused_color), dim=1)
            self._features_dc = nn.Parameter(features9channel.contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) #[n,1,3]
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) #[n,15,3]

        N, _ = fused_color.shape

        
        

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))


        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        

        self.motion_mlp.to('cuda')
        self.rot_mlp.to('cuda')
        self.hexplane.to('cuda')
        self.opacity_mlp.to('cuda')
        self.shs_mlp.to('cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #sigmoid time
        if all_frame_time_init:
            times = torch.tensor(np.asarray(pcd.times)).float().cuda()
        else:
            #use in the paper:N3d,dnerf
            times = torch.rand((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))


        self.mlp_grd = {}
        self.init_mlp_grd()

        # print(self._xyz.size())
        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2]) 
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1]) 
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0]) 
        
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets.. 
     
        if self.args.planemodel == "hexplane_mip":
            self.hexplane.set_base_scale(spatial_lr_scale)
    def static2dynamatic(self):
        self.is_dynamatic = True



    def cache_gradient(self,stage):
        '''把grad都传给grd'''
        if stage == "dynamatic":#只有dynamtic时，才会有下面这三个的梯度


            def add_mlp(mlp,mlp_name,weight_name=''):
                for name, W in mlp.named_parameters():
                    # if name =="grids.0.1":
                    #     print(self.batch_iter,name,W.grad)
                    if weight_name in name and W.grad is not None:
                        self.mlp_grd[mlp_name+name] += W.grad.clone()
            
            add_mlp(self.motion_mlp,"motion")
            add_mlp(self.rot_mlp,"rot")
            add_mlp(self.opacity_mlp,"opacity")
            add_mlp(self.hexplane,"hexplane","grids")
            add_mlp(self.shs_mlp,"shs")


            self._trbf_center_grd += self._trbf_center.grad.clone()




        self._xyz_grd += self._xyz.grad.clone()
        self._features_dc_grd += self._features_dc.grad.clone()
        self._scaling_grd += self._scaling.grad.clone()
        self._rotation_grd += self._rotation.grad.clone()
        self._opacity_grd += self._opacity.grad.clone()
        self._features_rest_grd += self._features_rest.grad.clone()


        
    def zero_gradient_cache(self):
        '''把grd都置零'''
        self._xyz_grd = torch.zeros_like(self._xyz, requires_grad=False)
        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._features_rest_grd = torch.zeros_like(self._features_rest, requires_grad=False)

        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        self._motion_grd = torch.zeros_like(self._motion, requires_grad=False)

        self._omega_grd = torch.zeros_like(self._omega, requires_grad=False)

        for name in self.mlp_grd.keys():
            self.mlp_grd[name].zero_()

    def set_batch_gradient(self, cnt,stage):

        '''把grd通过batch平均后传给grad'''
        ratio = 1/cnt
        self._features_dc.grad = self._features_dc_grd * ratio
        self._features_rest.grad = self._features_rest_grd * ratio
        self._xyz.grad = self._xyz_grd * ratio
        self._scaling.grad = self._scaling_grd * ratio
        self._rotation.grad = self._rotation_grd * ratio
        self._opacity.grad = self._opacity_grd * ratio
        self._trbf_center.grad = self._trbf_center_grd * ratio
        # print()
        t_nan = self._trbf_center.grad.isnan().sum()
        # print(t_nan)
        assert t_nan ==0
        self._motion.grad = self._motion_grd * ratio

        self._omega.grad = self._omega_grd * ratio


        if stage == "dynamatic":
            def set_mlp_gradient(mlp,mlp_name,weight_name=''):
                for name, W in mlp.named_parameters():                    
                    if weight_name in name :
                        W.grad = self.mlp_grd[mlp_name+name]*ratio

            set_mlp_gradient(self.motion_mlp,"motion")
            set_mlp_gradient(self.rot_mlp,"rot")
            set_mlp_gradient(self.opacity_mlp,"opacity")
            set_mlp_gradient(self.hexplane,"hexplane","grids")
            set_mlp_gradient(self.shs_mlp,"shs")
            
            # set_mlp_gradient(self.rgbdecoder,"rgbdecoder",'weight')
            # for name, W in self.motion_mlp.named_parameters():
            #     # if 'weight' in name:
            #         # print(name,W)
            #         W.grad = self.mlp_grd["motion"+name] * ratio
            # for name, W in self.rot_mlp.named_parameters():
            #     # if 'weight' in name:
            #         W.grad = self.mlp_grd["rot"+name] * ratio

            # for name, W in self.hexplane.named_parameters():
            #     if "grids" in name:
            #         W.grad = self.mlp_grd["hexplane"+name] * ratio

    def training_setup(self, training_args):
        '''设置optimizer'''
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.training_args = training_args
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.batch = training_args.batch
        l = [
        {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
        {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
        {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        {'params': list(self.motion_mlp.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "motion_mlp"},
        {'params': list(self.rot_mlp.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "rot_mlp"},
        {'params': list(self.opacity_mlp.parameters()), 'lr':   training_args.mlp_lr , "weight_decay":8e-7,"name": "opacity_mlp"},
        {'params': list(self.hexplane.parameters()), 'lr':   training_args.hexplane_lr*self.spatial_lr_scale , "weight_decay":8e-7,"name": "hexplane"},
        {'params': list(self.shs_mlp.parameters()), 'lr':   training_args.mlp_lr, "weight_decay":8e-7,"name": "shs_mlp"},
        {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
        ]

        

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15,fused=True)
        # print(len(self.optimizer.param_groups))
        # print(len(self.optimizer.state))
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_lr,
                                                    lr_final=training_args.mlp_lr_final,
                                                    # lr_delay_mult=training_args.mlp_lr_delay_mult,
                                                    # start_step = training_args.static_iteration,
                                                    start_step = -1,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deform_feature_scheduler_args = get_expon_lr_func(lr_init=training_args.deform_feature_lr,
                                                    lr_final=training_args.deform_feature_lr_final,
                                                    # lr_delay_mult=training_args.deform_feature_lr_delay_mult,
                                                    # start_step = training_args.static_iteration,
                                                    start_step = -1,
                                                    max_steps=training_args.position_lr_max_steps)
        self.hexplane_scheduler_args = get_expon_lr_func(lr_init=training_args.hexplane_lr,
                                                    lr_final=training_args.hexplane_lr_final,
                                                    # lr_delay_mult=training_args.hexplane_lr_delay_mult,
                                                    # start_step = training_args.static_iteration,
                                                    start_step = -1,
                                                    max_steps=training_args.position_lr_max_steps)  
        self.trbf_center_scheduler_args = get_expon_lr_func(lr_init=training_args.trbfc_lr,
                                                    lr_final=training_args.trbfc_lr_final,
                                                    # lr_delay_mult=training_args.hexplane_lr_delay_mult,
                                                    # start_step = training_args.static_iteration,
                                                    start_step = -1,
                                                    max_steps=training_args.position_lr_max_steps)
        self.inv_intergral =torch.ones_like(self._opacity)
        print("move decoder to cuda")

    def update_learning_rate(self, iteration,stage=None,use_intergral=True,scale_intergral=True):
        ''' Learning rate scheduling per step '''
        # print(self.inv_intergral,self.inv_intergral.max(),self.inv_intergral.min())
        if stage ==  "dynamatic" and iteration%50==0:
        #     # print(self.inv_intergral,self.inv_intergral.max(),self.inv_intergral.min())

            intergral = self.get_intergral()
            valid_mask = (intergral>self.min_intergral).squeeze()
            prune_mask = ~valid_mask
            self.prune_points(prune_mask)
            intergral =intergral[valid_mask]
            self.inv_intergral = 1/intergral

            self.inv_intergral = self.inv_intergral/self.inv_intergral.min()
            self.inv_intergral_fordensify = self.inv_intergral
            if not use_intergral:
                self.inv_intergral = torch.ones_like(self._opacity)

        if stage == "static":
            self.inv_intergral = torch.ones_like(self._opacity)
            self.inv_intergral_fordensify = self.inv_intergral

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                lr = lr*self.inv_intergral
                param_group['lr'] = lr

            elif "mlp" in param_group["name"] :
                lr = self.mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "motion" or param_group["name"] =="omega":
                lr = self.deform_feature_scheduler_args(iteration)
                # lr=0
                param_group['lr'] = lr
            elif param_group["name"] == "hexplane":
                lr = self.hexplane_scheduler_args(iteration)
                param_group['lr'] = lr

            elif param_group["name"] =="opacity":

                    lr = self.training_args.opacity_lr*self.inv_intergral

                    param_group['lr'] = lr
            elif param_group["name"] =="trbf_center":
                # lr = self.trbf_center_scheduler_args(iteration)* self.inv_intergral
                lr = self.training_args.trbfc_lr * self.inv_intergral
                param_group['lr'] = lr
                # print("trbf_center",lr)
            elif param_group["name"] == "scaling":
                lr = self.training_args.scaling_lr 
                if scale_intergral:
                    lr = lr * self.inv_intergral
                # * self.inv_intergral
                param_group['lr'] = lr
            elif param_group["name"] == "rotation":
                lr = self.training_args.rotation_lr * self.inv_intergral
                param_group['lr'] = lr
            elif param_group["name"] == "f_dc":
                lr = self.training_args.feature_lr * self.inv_intergral
                # param_group['lr'] = 0
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z' ,'nx', 'ny', 'nz']


        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('trbf_center')


        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        trbf_center= self._trbf_center.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, trbf_center, trbf_scale, normals, motion, f_dc, opacities, scale, rotation, omega), axis=1)
        attributes = np.concatenate((xyz, normals,f_dc,f_rest, opacities, scale, rotation,trbf_center), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pth")
        print(f'Saving model checkpoint to: {model_fname}')
        # torch.save(self.rgbdecoder.state_dict(), model_fname)
        mlp_dict = {'motion_state_dict': self.motion_mlp.state_dict(), 'rot_state_dict': self.rot_mlp.state_dict(),
        'hexplane_state_dict':self.hexplane,"opacity_state_dict":self.opacity_mlp.state_dict(),
        "shs_state_dict":self.shs_mlp.state_dict()}

        torch.save(mlp_dict, model_fname)



    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


   
   
    def load_ply(self, path):
        plydata = PlyData.read(path)
        #ckpt = torch.load(path.replace(".ply", ".pt"))
        #self.rgbdecoder.load_state_dict(ckpt)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])




        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])






        mlp_state_dict = torch.load(path.replace(".ply", ".pth"))




        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree


        self.motion_mlp.load_state_dict({k.replace('motion_', ''): v for k, v in mlp_state_dict['motion_state_dict'].items()})
        self.rot_mlp.load_state_dict({k.replace('rot_', ''): v for k, v in mlp_state_dict['rot_state_dict'].items()})
        # self.hexplane.load_state_dict({k.replace('hexplane_', ''): v for k, v in mlp_state_dict['hexplane_state_dict'].items()})
        self.opacity_mlp.load_state_dict({k.replace('opacity_', ''): v for k, v in mlp_state_dict['opacity_state_dict'].items()})
        self.shs_mlp.load_state_dict({k.replace('shs_', ''): v for k, v in mlp_state_dict['shs_state_dict'].items()})
        self.hexplane = mlp_state_dict["hexplane_state_dict"]
        
        self.motion_mlp.to("cuda")
        self.rot_mlp.to("cuda")
        self.hexplane.to("cuda")
        self.opacity_mlp.to("cuda")
        self.shs_mlp.to("cuda")

        self.mlp_grd = {}
        self.init_mlp_grd()

    def init_mlp_grd(self):
        def add_mlp(mlp,mlp_name,weight_name=''):
            for name, W in mlp.named_parameters():
                if weight_name in name:
                    self.mlp_grd[mlp_name+name] = torch.zeros_like(W, requires_grad=False).cuda()

        add_mlp(self.motion_mlp,"motion")
        add_mlp(self.rot_mlp,"rot")
        add_mlp(self.opacity_mlp,"opacity")
        add_mlp(self.hexplane,"hexplane","grids")
        add_mlp(self.shs_mlp,"shs")

    def replace_tensor_to_optimizer(self, tensor, name):
        '''将optim中对应name的值给换成tensor，并且adam中原本保存的状态清0'''
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # print(group,len(group["params"]))
            # print(group["name"],group["params"][0].shape,len(group["params"]))

            if len(group["params"]) == 1 and "mlp" not in group["name"] and group["name"] != "hexplane":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))) #生成的是leaf节点
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        '''将tensors cat 到optimizer中，即加入进去'''
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_feature_rest ,new_opacities, new_scaling, new_rotation, new_trbf_center, dummy=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_feature_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "trbf_center" : new_trbf_center,

        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d) #将这些点加入进去
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    



    def densify_and_splitv2(self, grads, grad_threshold, scene_extent, N=2,t_grads=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        #因为之前clone过了，所以要把grads给补齐
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()



        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        if self.args.rgbdecoder:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1)
        else:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1) # n,1,1 to n1
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacity, new_scaling, new_rotation,new_trbf_center)


        if self.args.enable_scale_sum:
            new_scale_sum = self.scale_sum[selected_pts_mask].repeat(N,1)/(0.8*N)
            self.scale_sum = torch.cat((self.scale_sum,new_scale_sum),dim=0)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    



    def densify_and_clone(self, grads, grad_threshold, scene_extent,t_grads=None):

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)


        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_trbf_center =  self._trbf_center[selected_pts_mask] # 
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacities, new_scaling, new_rotation,new_trbf_center)


        if self.args.enable_scale_sum:
            new_scale_sum = self.scale_sum[selected_pts_mask]
            self.scale_sum = torch.cat((self.scale_sum,new_scale_sum),dim=0)

    def densify_pruneclone(self, max_grad, min_opacity, extent, max_screen_size, splitN=1):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads = grads*self.inv_intergral_fordensify
        t_grads = self.t_gradient_accum / self.denom
        t_grads[t_grads.isnan()] = 0.0
        t_grads =None
        self.densify_and_clone(grads, max_grad, extent, t_grads)


        self.densify_and_splitv2(grads, max_grad, extent, 2, t_grads)


        prune_mask = (self.get_opacity < min_opacity).squeeze()

        intergral_mask = (self.get_intergral() <self.min_intergral).squeeze() #将intergral小于0的点给去掉

        prune_mask = torch.logical_or(prune_mask, intergral_mask)
        if self.args.loader == "colmap":
            z_mask = (self.get_xyz[:,2] < 4.5).squeeze()
            # print("pure_z",z_mask.sum())
            prune_mask = torch.logical_or(prune_mask, z_mask)#把这个移到外面去


        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # spatial_scale = self.get_real_scale()  
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            if self.args.pw:
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            else:
                prune_mask = torch.logical_or(prune_mask,  big_points_vs)#只用vs
        self.prune_points(prune_mask)

        if self.args.enable_scale_sum:
            self.init_real_scale()
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        # if self.gaussian_dim == 4:
        if avg_t_grad is not None:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

   

    def prune_pointswithemsmask(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        #self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.maskforems = self.maskforems[valid_points_mask] # we only remain valid mask from ems 



    def reducepointsopscalebymask(self, selected_pts_mask):
        # restrict orignal one
        meanvalue = torch.min(self.get_opacity)
        opacities_new = inverse_sigmoid( torch.min( torch.ones_like(self.get_opacity[selected_pts_mask])*meanvalue , torch.ones_like(self.get_opacity[selected_pts_mask])*0.004))
        opacityold = self._opacity.clone()
        opacityold[selected_pts_mask] = opacities_new


        optimizable_tensors = self.replace_tensor_to_optimizer(opacityold, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def get_motion(self, timestamp):
        #给visualize用的
        time_embbed = self.time_emb(torch.tensor([timestamp],dtype=torch.float, device="cuda"))
        # print(time_embbed)
        time_embbed = time_embbed.repeat(self._motion.shape[0],1)
        motion_input = torch.cat((self._motion,time_embbed),dim=1)

        residual = self.motion_mlp(motion_input)
        motion = self._xyz.detach() + residual

        return motion



    @property
    def get_points_num(self):
        return self._xyz.shape[0]

    def get_trbfoutput(self,trbfdistance):
        return torch.exp(-4*(trbfdistance**2))
    
    def get_intergral(self,start=0.0,end=1.0):
        # print(self.get_trbfscale.shape,self.get_trbfcenter.shape)

        with torch.no_grad():
            hexplane_feature = self.hexplane(self._xyz.detach(),self.get_trbfcenter.detach(),self.get_scaling.detach()) #[N,D]
            trbf_scale = 1-self.opacity_mlp(hexplane_feature.clone()) #得到trbf_scale
            min_scale = self.args.min_interval/(self.duration)
            trbf_scale = (1-min_scale)*trbf_scale + min_scale #限制min_scale最小值
        # 计算output函数在start和end之间的积分
        def Q(x:torch.Tensor):
            # print(x)
            a1 = torch.tensor([0.070565902],device="cuda")
            a2 = torch.tensor([1.5976],device="cuda") 
            # print(a1*x**3+a2*x)
            return (1-1/(1+torch.exp(a1*x**3+a2*x)))
        p1 = Q(2*np.sqrt(2)*(end-self.get_trbfcenter)/trbf_scale)
        p2 = Q(2*np.sqrt(2)*(start-self.get_trbfcenter)/trbf_scale)
        # print(p1,p2)
        return trbf_scale*np.sqrt(np.pi)/2*(p1-p2)

    def get_deformation(self,timestamp,rays=None):
        scales = torch.cat((self.get_scaling.detach(),self._trbf_scale.detach()/2),dim=1)
        hexplane_feature = self.hexplane(self._xyz.detach(),self.get_trbfcenter.detach(),scales.detach()) #[N,D]

        trbf_scale = 1-self.opacity_mlp(hexplane_feature) #得到trbf_scale
        min_scale = self.args.min_interval/(self.duration)
        trbf_scale = (1-min_scale)*trbf_scale + min_scale #限制min_scale最小值
        self._trbf_scale = trbf_scale


        trbfdistanceoffset = timestamp  - self.get_trbfcenter
        trbfdistance =  trbfdistanceoffset / trbf_scale
        trbfoutput = self.get_trbfoutput(trbfdistance)

        time_embbed = self.time_emb(trbfdistanceoffset)
        deform_feature = torch.cat((hexplane_feature,time_embbed.detach()),dim=1)
        
        base_time_embbed = self.time_emb(torch.zeros_like(trbfdistanceoffset))
        base_deform_feature = torch.cat((hexplane_feature,base_time_embbed.detach()),dim=1)
        if self.args.scale_reg:
            self.scale_residual = self.rot_mlp(base_deform_feature)[:,4:]
        if self.args.shs_reg:
            self.shs_residual = self.shs_mlp(base_deform_feature).reshape(-1,16,3)
        if self.args.motion_reg:
            self.motion_residual = self.motion_mlp(base_deform_feature)

        with torch.no_grad():
            self.real_xyz = self._xyz + self.motion_mlp(base_deform_feature)


        if self.args.dx:
            motion_residual = self.motion_mlp(deform_feature)
            motion = self._xyz + motion_residual
        else:
            motion = self._xyz
        
        if self.args.drot:
            rot_residual = self.rot_mlp(deform_feature)

            base_scale = self.get_scaling.detach()

            rot = self._rotation + rot_residual[:,:4]


            rot = self.rotation_activation(rot)

            if self.args.scale_rot:
                scale_res =  rot_residual[:,4:]
                scale = self._scaling + rot_residual[:,4:]
                scale = self.scaling_activation(scale)
            
        else:
            rot = self.get_rotation
            if self.args.scale_rot:
                scale = self.get_scaling
        #若为scale_rot,则scale归上面的管。否则scale归下面的管
        if not self.args.scale_rot:
            if self.args.dscale:
                scale_res = self.scale_mlp(deform_feature)
                scale = self._scaling + scale_res
                scale = self.scaling_activation(scale)
            else:
                scale = self.get_scaling

        if self.args.dopacity:

            opacity = self._opacity
            opacity = self.opacity_activation(opacity)*trbfoutput

        else:
            opacity = self.get_opacity

        if self.args.dsh:
            shs_residual = self.shs_mlp(deform_feature).reshape(-1,16,3)
            features_dc =  self._features_dc 
            features_rest = self._features_rest
            shs = torch.cat((features_dc, features_rest), dim=1) + shs_residual
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
            shs =  torch.cat((features_dc, features_rest), dim=1)


        return motion, rot,scale,opacity,shs,None
    



    def set_bounds(self,xyz_max, xyz_min):
        print("set bounds")
        bounds = torch.tensor([xyz_max, xyz_min],dtype=torch.float32,device='cuda')
        print(bounds)
        self.bounds = nn.Parameter(bounds,requires_grad=False)
        self.hexplane.set_aabb(xyz_max,xyz_min,self.duration)
    def inv_normalize_xyz(self,norm_xyz):
        min_bounds = self.bounds[1]
        max_bounds = self.bounds[0]
        return norm_xyz * (max_bounds - min_bounds) + min_bounds
    def normalize_residual(self,norm_xyz):
        '''0-1 -> -(max_bounds - min_bounds) - (max_bounds - min_bounds)'''
        return (2*norm_xyz-1) * (self.bounds[0] - self.bounds[1])
    def get_deformfeature(self):
        scales = torch.cat((self.get_scaling.detach(),torch.zeros((self.get_scaling.shape[0],1),device="cuda")),dim=1)

        # scales = torch.cat((self.get_scaling.detach(),self._trbf_scale.detach()/2),dim=1)
        # print(scales.shape)
        self.hexplane_feature = self.hexplane(self._xyz.detach(),self.get_trbfcenter.detach(),scales.detach()) #[N,D]
        trbf_scale = 1-self.opacity_mlp(self.hexplane_feature) #得到trbf_scale
        min_scale = self.args.min_interval/(self.duration)
        trbf_scale = (1-min_scale)*trbf_scale + min_scale #限制min_scale最小值
        self._trbf_scale = trbf_scale
    def get_deformation_eval(self,timestamp,rays=None):
        trbfdistanceoffset = timestamp  - self.get_trbfcenter
        trbfdistance =  trbfdistanceoffset / self._trbf_scale
        trbfoutput = self.get_trbfoutput(trbfdistance)

        time_embbed = self.time_emb(trbfdistanceoffset)
        deform_feature = torch.cat((self.hexplane_feature,time_embbed.detach()),dim=1)
        
        select_mask = trbfoutput>0.001

        deform_feature = deform_feature[select_mask.squeeze()]
        trbfoutput = trbfoutput[select_mask.squeeze()]
        if self.args.onemlp:
            motion_residual  = self.motion_mlp(deform_feature)
            motion = self._xyz + motion_residual[:,:3]
            rot = self._rotation + motion_residual[:,3:7]
            rot = self.rotation_activation(rot)

            scale = self._scaling + motion_residual[:,7:]
            scale = self.scaling_activation(scale)
        else:
            if self.args.dx:

                motion_residual = self.motion_mlp(deform_feature)

                motion = self._xyz[select_mask.squeeze()] + motion_residual


            else:
                motion = self._xyz
            
            if self.args.drot:
                rot_residual = self.rot_mlp(deform_feature)
                rot = self._rotation[select_mask.squeeze()] + rot_residual[:,:4]

                rot = self.rotation_activation(rot)

                if self.args.scale_rot:
                    scale_res =  rot_residual[:,4:]
                    scale = self._scaling[select_mask.squeeze()] + rot_residual[:,4:]
                    scale = self.scaling_activation(scale)
            else:
                rot = self.get_rotation
                if self.args.scale_rot:
                    scale = self.get_scaling
            #若为scale_rot,则scale归上面的管。否则scale归下面的管
            if not self.args.scale_rot:
                if self.args.dscale:
                    scale_res = self.scale_mlp(deform_feature)
                    scale = self._scaling + scale_res
                    scale = self.scaling_activation(scale)
                else:
                    scale = self.get_scaling

        if self.args.dopacity:

            opacity = self._opacity[select_mask.squeeze()]
            opacity = self.opacity_activation(opacity)*trbfoutput

        else:
            opacity = self.get_opacity

        if not self.args.rgbdecoder:
            if self.args.dsh:
                shs_residual = self.shs_mlp(deform_feature).reshape(-1,16,3)
                # print(shs_residual)
                features_dc =  self._features_dc[select_mask.squeeze()] 
                features_rest = self._features_rest[select_mask.squeeze()]
                shs = torch.cat((features_dc, features_rest), dim=1) + shs_residual
            else:
                features_dc = self._features_dc
                features_rest = self._features_rest
                shs =  torch.cat((features_dc, features_rest), dim=1)
            # print(self.hexplane.aabb)

        if self.args.rgbdecoder:
            rgb_feature = self.shs_mlp(deform_feature)+self._features_dc
            shs=None
        else:
            rgb_feature = None
        return motion, rot,scale,opacity,shs,rgb_feature
def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : i,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2'] #L-1
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)