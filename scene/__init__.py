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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams
from PIL import Image 
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_camInfosv2, cameraList_from_camInfosv2nogt
from utils.system_utils import mkdir_p
from helper_train import recordpointshelper
import torch 
from scene.dataset import CameraDataset
class Scene:

    # gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], multiview=False,duration=50.0, loader="colmap", is_rendering = False):
        """
        :param path: Path to colmap scene main folder.
        """
        self.args=args
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background
        self.refmodelpath = None
        self.loader = loader
        self.all_frame_time_init = False

        if load_iteration:
            self.loaded_iter = load_iteration
            # if load_iteration == -1:
            #     self.loaded_iter = "best"
            #     # self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            # else:
            #     self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.val_cameras = {}
        raydict = {}


        if loader == "colmap" : # colmapvalid only for testing
            self.scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, multiview, duration=duration)
        elif loader == "blender" :
            self.scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, multiview=multiview,duration=duration)
        else:
            assert False, "Could not recognize scene type!"

        xyz_max = self.scene_info.point_cloud.points.max(axis=0)
        xyz_min = self.scene_info.point_cloud.points.min(axis=0)
        self.gaussians.set_bounds(xyz_max, xyz_min)

        if not self.loaded_iter:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=2)

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling

 
        self.cameras_extent = self.scene_info.nerf_normalization["radius"] 
        print("radius:",self.cameras_extent)


        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")  
            if is_rendering:
            # if loader in ["colmapvalid","blendervalid"]:         
                self.train_cameras[resolution_scale] = [] # no training data


            else: # n3d dnerf basketball
                self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(self.scene_info.train_cameras, resolution_scale, args)
            
            
            
            print("Loading Test Cameras")
            # if loader  in ["colmapvalid",  "colmap","blender","blendervalid"]: # we need gt for metrics
            self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(self.scene_info.test_cameras, resolution_scale, args)
            # else:
            #     raise NotImplementedError
            print("Loading Val Cameras")
            if loader in ["colmap"] and is_rendering:
                self.val_cameras[resolution_scale] = cameraList_from_camInfosv2nogt(self.scene_info.val_cameras, resolution_scale, args)




        if self.loaded_iter :
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent,all_frame_time_init = self.all_frame_time_init)


    # def save(self, iteration):
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    def save(self, iteration,best_ckpt=False):
        if best_ckpt:
            save_path = os.path.join(self.model_path, "point_cloud/best_ckpt.ply".format(iteration))
        else:
            save_path = os.path.join(self.model_path, "point_cloud/iteration_{}.ply".format(iteration))
        mkdir_p(os.path.dirname(save_path))
        self.gaussians.save_ply(save_path)

    def recordpoints(self, iteration, string):
        txtpath = os.path.join(self.model_path, "exp_log.txt")
        numpoints = self.gaussians._xyz.shape[0]
        recordpointshelper(self.model_path, numpoints, iteration, string)

    def getTrainCameras(self, scale=1.0):
        # 因为图片数量过多，不能一次加载到cuda或cpu memory中，
        if self.args.use_loader:
            use_background = False
            if self.loader in ["blender","blendervalid"]:
                use_background = True
            return CameraDataset(self.train_cameras[scale].copy(),self.white_background,use_background=use_background)#到这里才把image转成了torch，之前一直用的numpy
        else:
            return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        if self.args.use_loader:
            use_background = False
            if self.loader in ["blender","blendervalid"]:
                use_background = True
            return CameraDataset(self.test_cameras[scale].copy(),self.white_background,use_background=use_background)
        return self.test_cameras[scale]
    
    def getValCameras(self, scale=1.0):
        if self.args.use_loader:
            use_background = False
            if self.loader in ["blender","blendervalid"]:
                use_background = True
            return CameraDataset(self.val_cameras[scale].copy(),self.white_background,use_background=use_background)
        return self.val_cameras[scale]

    def getTrainCamInfos(self):
        # if self.args.use_loader:
        #     return CameraDataset(self.scene_info.train_cameras.copy())
        return self.scene_info.train_cameras
 