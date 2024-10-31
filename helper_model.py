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

import torch 
from thirdparty.gaussian_splatting.utils.graphics_utils import BasicPointCloud
import numpy as np
from simple_knn._C import distCUDA2
from mmcv.ops import knn
import torch.nn as nn



class Sandwich(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwich, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) # 

        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = self.sigmoid(result) 
        return result


class Sandwichnoact(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwichnoact, self).__init__()
        
        # self.mlp2 = nn.Conv2d(11, 6, kernel_size=1, bias=bias) # double hidden layer..
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) # double hidden layer..

        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()



    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = torch.clamp(result, min=0.0, max=1.0)
        return result
####### following are also good rgb model but not used in the paper, slower than sandwich, inspired by color shift in hyperreel
# remove sigmoid for immersive dataset
class RGBDecoderVRayShift(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(RGBDecoderVRayShift, self).__init__()
        
        self.mlp1 = nn.Conv2d(dim, outdim, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(15, outdim, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(6, outdim, kernel_size=1, bias=bias)
        self.sigmoid = torch.nn.Sigmoid()

        self.dwconv1 = nn.Conv2d(9, 9, kernel_size=1, bias=bias)

    def forward(self, input, rays, t=None):
        x = self.dwconv1(input) + input 
        albeado = self.mlp1(x)
        specualr = torch.cat([x, rays], dim=1)
        specualr = self.mlp2(specualr)

        finalfeature = torch.cat([albeado, specualr], dim=1)
        result = self.mlp3(finalfeature)
        result = self.sigmoid(result)   
        return result 
    
class myrgb(nn.Module):
    #输入是feature（带时间的）和ray
    def __init__(self, dim=9,dir_dim=6, outdim=3, bias=False):
        super(myrgb, self).__init__()
        
        self.mlp1 = nn.Conv2d(dim, outdim, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(dim+dir_dim, outdim, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(6, outdim, kernel_size=1, bias=bias)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, rays, t=None):
        albeado = self.mlp1(input)
        specualr = torch.cat([input, rays], dim=1)
        specualr = self.mlp2(specualr)

        finalfeature = torch.cat([albeado, specualr], dim=1)
        result = self.mlp3(finalfeature)
        result = self.sigmoid(result)   
        return result

def interpolate_point(pcd, N=4):
    #对点进行过滤，让其更稀疏。N越大过滤的越多。
    save_rate = 1.0/N
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)


    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time # mask的是 时间都为oldtime的点
        selectedmask = selectedmask.squeeze(1)
        
        if timeidx == 0:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])
        else:
            xyzinput = oldxyz[selectedmask] #找出同一个时间的点
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3
            xyznnpoints = knn(2, xyzinput, xyzinput, False)

            nearestneibourindx = xyznnpoints[0, 1].long() # N x 1   
            spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance) 
            N = spatialdistance.shape[0]
            num_take = int(N * save_rate)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selectedmask][masksnumpy])
            newcolor.append(oldcolor[selectedmask][masksnumpy])
            newnormal.append(oldnormal[selectedmask][masksnumpy])
            newtime.append(oldtime[selectedmask][masksnumpy])
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  


    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd



def interpolate_pointv3(pcd, N=4,m=0.25):
    
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)


    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        if timeidx % N == 0:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

        else:
            xyzinput = oldxyz[selectedmask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3
            xyznnpoints = knn(2, xyzinput, xyzinput, False)

            nearestneibourindx = xyznnpoints[0, 1].long() # N x 1  skip the first one, we select the second closest one
            spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance) 
            M = spatialdistance.shape[0]
            num_take = int(M * m)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selectedmask][masksnumpy])
            newcolor.append(oldcolor[selectedmask][masksnumpy])
            newnormal.append(oldnormal[selectedmask][masksnumpy])
            newtime.append(oldtime[selectedmask][masksnumpy])
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  


    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd




def interpolate_partuse(pcd, N=4):
    # used in ablation study
    #这个是隔多少帧取一次点
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)

    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        if timeidx % N == 0:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

        else:
            pass
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  

    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd

def prune_point(pcd, maxz=200):
    #去除高度大于maxz的点
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    selectedmask = oldxyz[:,2] < maxz
    newxyz = oldxyz[selectedmask]
    newcolor = oldcolor[selectedmask]
    # newnormal = oldnormal[selectedmask]
    newtime = oldtime[selectedmask]
    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)
    return newpcd
def add_extra_point(pcd,extra_point_num=5000,radius=200,min_radius=63):
    #针对coffee右侧窗户，增加点
    # extra_point_num=5000
    max = np.max(pcd.points, axis=0)
    min = np.min(pcd.points, axis=0)
    print(max)
    phi_max = np.arctan(max[2]/max[1])
    print(phi_max)
    sita_max =np.pi/4
    radius = np.random.rand(extra_point_num)*radius+min_radius
    phi = np.random.rand(extra_point_num)*np.pi/2 + np.pi/4
    sita = np.random.rand(extra_point_num)*sita_max
    x= radius*np.sin(phi)*np.cos(sita)
    y=radius*np.cos(phi)
    # print("phi_max",phi.max())
    # print("phi_min",phi.min())
    # print("cosphi",np.cos(phi).max())
    # print("ymax",y.max())
    z=radius*np.sin(phi)*np.sin(sita)
    xyz_extra = np.stack([x, y, z], axis=1)
    # normals_extra = np.zeros_like(xyz_extra)
    rgb_extra = np.ones((extra_point_num, 3)) / 2
    times_extra = np.ones((extra_point_num, 1)) * 0.5
    newxyz = np.concatenate([pcd.points, xyz_extra], axis=0)
    newcolor = np.concatenate([pcd.colors, rgb_extra], axis=0)
    # newnormal = np.concatenate([pcd.normals, normals_extra], axis=0)
    newtime = np.concatenate([pcd.times, times_extra], axis=0)
    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)
    return newpcd
def padding_point(pcd, N=4):
    
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times
    
    timestamps = np.unique(oldtime)
    totallength = len(timestamps)


    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selectedmask = oldtime == time
        selectedmask = selectedmask.squeeze(1)
        
        

        if timeidx != 0 and timeidx != len(timestamps) - 1:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

             
        else:
            newxyz.append(oldxyz[selectedmask])
            newcolor.append(oldcolor[selectedmask])
            newnormal.append(oldnormal[selectedmask])
            newtime.append(oldtime[selectedmask])

            xyzinput = oldxyz[selectedmask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous() # 1 x N x 3


            xyznnpoints = knn(2, xyzinput, xyzinput, False)


            nearestneibourindx = xyznnpoints[0, 1].long() # N x 1  skip the first one, we select the second closest one
            spatialdistance = torch.norm(xyzinput - xyzinput[:,nearestneibourindx,:], dim=2) #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance) 
            N = spatialdistance.shape[0]
            num_take = int(N * 0.125)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selectedmask][masksnumpy])
            newcolor.append(oldcolor[selectedmask][masksnumpy])
            newnormal.append(oldnormal[selectedmask][masksnumpy])

            if timeidx == 0:
                newtime.append(oldtime[selectedmask][masksnumpy] - (1 / totallength)) 
            else :
                newtime.append(oldtime[selectedmask][masksnumpy] + (1 / totallength))
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]  


    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd

def getcolormodel(rgbfuntion,feature_dim=9,dir_out_dim=6):
    if rgbfuntion == "sandwich":
        rgbdecoder = Sandwich(9,3)
    
    elif rgbfuntion == "sandwichnoact":
        rgbdecoder = Sandwichnoact(9,3)
    elif rgbfuntion =="myrgb":
        rgbdecoder = myrgb(feature_dim,dir_out_dim)
    else :
        return None 
    return rgbdecoder

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0


def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5



