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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self._test_cam_idx = []
        self.data_device = "cuda"
        self.veryrify_llff = 0
        self.eval = True #是否要将数据集分为train和test
        self.model = "gmodel" # 
        self.loader = "colmap" #
        self.use_loader = False #是否使用dataloader加载数据。适用于n3d这种多视角帧数比较多的数据集
        self.color_order = -1
        self.random_init = False
        self.deform_feature_dim = 16
        self.deform_hidden_dim = 128
        self.deform_time_encode =4

        self.dx =True
        self.drot = True
        self.dscale = False
        self.dopacity = True
        self.dsh = False
        self.scale_rot = True #false表示将scale rot分开学习，反之表示一起学习
        self.dynamatic_mlp =False
        self.key_frame_nums = -1
        self.use_shs =True
        self.scale_reg = False
        self.shs_reg = False
        self.motion_reg = False
        self.all_frame_time_init =False
        #hexplane config:
        self.bounds = 1.6
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }
        self.multires = [1, 2, 4, 8]

        self.fine_kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }
        self.fine_multires = [1, 2, 4, 8]

        self.abtest1=False
        self.multiscale_time = False
        self.time_batch = False
        self.pre_activ = True
        self.planemodel = "hexplane"
        self.min_intergral = 0.1
        self.min_interval =1
        self.enable_coarse2fine = False
        self.enable_scale_sum = False
        self.rgbdecoder =None
        self.onemlp =False
        # self.add_points = False
        self.sigmoid_tcenter = False
        self.pw=False
        super().__init__(parser, "Loading Parameters", sentinel) #sentinel为true表示不采用类里定义的默认值，将默认值统一设为None

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.featuret_lr = 0.001
        self.opacity_lr = 0.05 #初始是0.05
        self.scaling_lr = 0.005

        self.trbfc_lr = 0.0001 # 
        self.trbfc_lr_final = 0.0000001
        self.trbfs_lr = 0.03
        self.trbfslinit = 0.0 # 
        self.batch = 2
        self.movelr = 4e-4
        self.dddm_lr = 1.6e-3

        self.deform_feature_lr = 1.6e-4
        self.deform_feature_lr_final = 1.6e-7

        self.mlp_lr = 1.6e-4
        self.mlp_lr_init = 1.6e-4
        self.mlp_lr_final = 1.6e-7

        self.hexplane_lr = 3.2e-3
        self.hexplane_lr_final = 3.2e-6

        self.start_lr = 1e-4
        self.omega_lr = 4e-4
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dtstd = 0
        self.lambda_dscale_entropy = 0
        self.lambda_dl1_opacity = 0
        self.lambda_dscale_reg = 0
        self.lambda_dshs_reg = 0
        self.lambda_dmotion_reg = 0
        self.lambda_dplanetv = 0
        self.lambda_dtime_smooth = 0
        self.densification_interval = 100
        self.opacity_reset_interval = 3_000
        self.opacity_reset_at = 10000
        self.densify_from_iter = 500
        self.densify_until_iter = 9000
        self.densify_grad_threshold = 0.0002
        self.rgb_lr = 0.0001
        self.desicnt = 6
        self.reg = 0 
        self.regl = 0.01 #原来是 1e-4
        self.shrinkscale = 2.0 
        self.randomfeature = 0 
        self.emstype = 0
        self.radials = 10.0
        self.farray = 2 # 
        self.emsstart = 1500 #small for debug,暂时先不用
        self.losstart = 200
        self.saveemppoints = 0 #
        self.prunebysize = 0 
        self.emsthr = 0.6  
        self.opthr = 0.005
        self.selectiveview = 0  
        self.preprocesspoints = 40  
        self.fzrotit = 8001
        self.addsphpointsscale = 0.8  
        self.gnumlimit = 330000 
        self.rayends = 7.5
        self.raystart = 0.7
        self.shuffleems = 1
        self.prevpath = "1"
        self.loadall = 0
        self.removescale = 5

        self.multiview = True
        self.sametime_batch = True
        self.keyindex_batch = False #目前默认用上面那个
        self.static_iteration = -1 # -1 静态场景训练结束时间。-1表示直接就是动态的
        self.use_weight_decay = False
        self.max_points_num = -1
        
        self.use_intergral_afterdensify = True
        self.all_no_intergral = False
        #hexplane config:
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001

        self.coarse_iteration = -1
        self.coarse_opacity_reset_interval = -1
        self.pure_static = -1
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    print(merged_dict)
    print("cmd_line",args_cmdline)
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
