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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
import glob
import natsort
from simple_knn._C import distCUDA2
import torch
from scene.hyper_loader import Load_hyper_data, format_hyper_data
from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int #对应的哪个相机
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str #照片路径
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float #照片对应的时间
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float
    # mask: np.array #hypernerf会用

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    val_cameras: list
    nerf_normalization: dict
    ply_path: str
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []
    # pose in llff. pipeline by hypereel 
    originnumpy = os.path.join(os.path.dirname(os.path.dirname(images_folder)), "poses_bounds.npy")
    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)


        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_bounds[:, -2:]


        near = bounds.min() * 0.95
        far = bounds.max() * 1.05
        
        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # 19, 3, 5


        val_poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        print(val_poses[0])
        val_pose = get_spiral(val_poses,near,far,N_views=300)
        print(val_pose.shape)


        H, W, focal = poses[0, :, -1]
        cx, cy = W / 2.0, H / 2.0

        K = np.eye(3)
        K[0, 0] = focal * W / W / 2.0
        K[0, 2] = cx * W / W / 2.0
        K[1, 1] = focal * H / H / 2.0
        K[1, 2] = cy * H / H / 2.0
        
        imageH = int (H//2) # note hard coded to half of the original image size
        imageW = int (W//2)
      
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number
     
    print(near,far)
    # c2w_list = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]




        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        # print(height,width)
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"



        for j in range(startime, startime+ int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0] #such as cam_00
            image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path)

            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1,cxr=0.0, cyr=0.0)

            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    val_caminfos = format_render_poses(val_pose,FovX,FovY,width,height,near,far)


    return cam_infos,val_caminfos
def format_render_poses(poses,FovX,FovY,width,height,near,far):
    cameras = []
    # tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    # image = data_infos[0][0]
    image = None
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        # R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        # FovX = focal2fov(data_infos.focal[0], image.shape[2])
        # FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=time, pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)

        cameras.append(cam_info)

    return cameras
def get_spiral(c2ws_all, near,far, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    # close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    close_depth, inf_depth = near * 0.9, far* 5.0
    print("focal",dt, close_depth, inf_depth, near, far)
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])


    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:

        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )

        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))

        render_poses.append(viewmatrix(z, up, c))
    return render_poses
def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg
def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m





def normalize(v):
    return v / np.linalg.norm(v)












def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    times = np.vstack([vertices['t']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)

def fetchPly3D(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals,times=None)

def storePly(path, xyzt, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('t','f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def storePly3D(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)






def readColmapSceneInfo(path, images, eval, llffhold=8, multiview=False, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file) 
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    parentdir = os.path.dirname(path)

    near = 0.01
    far = 100

    starttime = os.path.basename(path).split("_")[1] # colmap_0, 
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    starttime = int(starttime)


    cam_infos_unsorted,val_cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), near=near, far=far, startime=starttime, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)


    if eval:
        train_cam_infos =  cam_infos[duration:] 
        test_cam_infos = cam_infos[:duration]#第一个cam下的作为test
        uniquecheck = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in uniquecheck:
                uniquecheck.append(cam_info.image_name)
        assert len(uniquecheck) == 1 
        
        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanitycheck:
                sanitycheck.append(cam_info.image_name)
        for testname in uniquecheck:
            assert testname not in sanitycheck
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:2] #dummy

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    duration =300
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")

    
    if not os.path.exists(totalply_path):
        #将所有时间的点云进行融合
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        totalxyz = []
        totalrgb = []
        totaltime = []
        for i in range(starttime, starttime + duration):
            thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_"+ str(starttime), "colmap_" + str(i), 1)
            xyz, rgb, _ = read_points3D_binary(thisbin_path)
            # print(xyz.shape)
            # print(xyz.max(axis=0))
            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(np.ones((xyz.shape[0], 1)) * (i-starttime) / duration)
        xyz = np.concatenate(totalxyz, axis=0)
        # print(len(xyz))
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        assert xyz.shape[0] == rgb.shape[0]  
        xyzt =np.concatenate( (xyz, totaltime), axis=1)     
        storePly(totalply_path, xyzt, rgb)
    try:
        pcd = fetchPly(totalply_path)
        print(pcd.points.shape)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           val_cameras = val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path)
    return scene_info


    









def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png",duration=None):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # print(frame["file_path"] ,extension)
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            
            timestamp =  frame.get('time', 0.0) * (duration-1)/duration
            # for j in range(20):
            # print(timestamp)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,timestamp=timestamp,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],near=None,far=None,pose=None,hpdirecitons=None,cxr=0.0,cyr=0.0))
            

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", multiview=False,duration=None):
    print("Reading Training Transforms")
    # print(extension)
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension,duration)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,duration)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        times = np.random.random((num_pts, 1))
        xyzt =np.concatenate( (xyz, times), axis=1)
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)),times=times)

        storePly(ply_path, xyzt, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           val_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    # print(type(cam_list))
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo, 
}


