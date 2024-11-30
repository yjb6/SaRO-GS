import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    # print("pts",pts.max(dim=0),pts.min(dim=0))
    return (pts - aabb[0])  / (aabb[1] - aabb[0])

def normalize_time(time,duration):
    # print("time",time.max(),time.min())
    return time*duration/(duration-1)

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor,levels,max_level=None, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    B, feature_dim = grid.shape[:2]
    grid = grid.permute(0,2,3,1).contiguous()
    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))#[B,1,N,2]

    n = coords.shape[-2]
    # print(coords,grid.shape)
    # interp = grid_sampler(
    #     grid,  # [B, feature_dim, reso, ...]
    #     coords,  # [B, 1, ..., n, grid_dim]
    #     align_corners=align_corners,
    #     mode='bilinear', padding_mode='border')
    levels=torch.min(levels,dim=-1).values.unsqueeze(0).unsqueeze(0)#取他们两者之间最小的level
    # print(levels)
    # levels = torch.zeros_like(levels)
    # print(grid.shape)
    # print("level",levels.min(),levels.max())
    # print("coord",coords.max(),coords.min())
    # print(levels.shape)
    interp =nvdiffrast.torch.texture(
            grid,
            coords,
            mip_level_bias=levels,
            # filter_mode="linear-mipmap-nearest",
            boundary_mode="clamp",
            max_mip_level=7,
        )  # 3xNx1xC
    # print(interp.shape)
    interp = interp.view(B, n,feature_dim)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 1):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd)) #在range(indim)中得到长度为grid_nd的组合数
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.zeros_(new_grid_coef)
            # nn.init.ones_(new_grid_coef)
            # nn.init.uniform_(new_grid_coef, a=a, b=b)

        else:
            # nn.init.uniform_(new_grid_coef, a=a, b=b)
            nn.init.zeros_(new_grid_coef)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            levels,
                            num_levels: Optional[int],
                            concat_planes: bool = False,

                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = []
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb],levels[...,coo_comb],)
                .view(-1, feature_dim)
            )
            # if 3 in coo_comb:
            #     print("levelmax",levels[...,coo_comb].min(dim=-1).values.mean())
            # compute product over planes
            # interp_space = interp_space + interp_out_plane
            interp_space.append(interp_out_plane)

        # print(interp_space)
        # combine over scales
        if concat_planes:
            # interp_space = torch.cat(interp_space, dim=-1)
            space = interp_space[0] + interp_space[1] + interp_space[3]
            time = interp_space[2] + interp_space[4] + interp_space[5]
            interp_space = torch.cat((space,time),dim=-1)
        else:
            interp_space = sum(interp_space)
            # print(interp_space.shape)
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()

def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h-1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w-1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg

class HexPlaneField_mip_allscale(nn.Module):
    '''
    针对rebuttle中提到的spatialtemporal平面也要采用scale-aware策略的实验
    '''
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        self.concat_plane = False#若为false，向量之间就是加和，若为true，就是拼接
        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        self.reso_list=[]
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            #ablation study
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            self.reso_list.append(config["resolution"])
            # config["resolution"] = [
            #     r * res for r in config["resolution"][:4]
            # ] 
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        if self.concat_plane:
            self.feat_dim = self.feat_dim * 2
        print("feature_dim:",self.feat_dim)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min,duration):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32,device="cuda")
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        self.duration =duration

        print("Voxel Plane: set aabb=",self.aabb)

        self.per_grid_size=[]
        for res in self.reso_list:
            per_dim_list=[]
            for i in range(3):
                per_dim_list.append((xyz_max[i]-xyz_min[i])/res[i])
            per_dim_list.append(1/res[3])
            self.per_grid_size.append(per_dim_list)
        self.max_level = torch.log(torch.tensor(self.reso_list[0]))
        self.base_scale = torch.tensor(self.per_grid_size[0],dtype=torch.float32,device="cuda")
    
    def get_level(self,scales:torch.Tensor):
        #算出每个维度的level
        min_scale = self.base_scale/2
        # print(min_scale)
        max_scale = min_scale*torch.tensor(self.reso_list[0]).to(min_scale)
        # print(min_scale,max_scale)
        # print(scales)
        # print("min_scale",min_scale)
        # print("scales",scales)
        scales = torch.clamp(scales,min_scale,max_scale)
        # print(scales)
        level = torch.log2(2*scales / self.base_scale.unsqueeze(0)) #[N,4]
        # print(level.shape,level)
        # print(level)
        # print(level.mean(dim=0),level.max(dim=0),level.min(dim=0))
        # level = scales/self.base_scale
        # level[:,3] = 0.0
        # level = torch.zeros_like(level)
        # level.to(torch.float32)
        # print("11",level,level.shape)
        return level

    def set_base_scale(self,scale):
        pass
        # scale = 0.1*scale
        # print(scale)
        # self.base_scale = torch.tensor([scale,scale,scale,1],device="cuda",dtype=torch.float32)
        # self.max_level = 1
    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None,scales: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb) 
        timestamps = normalize_time(timestamps, self.duration) #[0,(d-1)/d] ->[-1,1]
        # print(timestamps)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        level = self.get_level(scales)
        pts = pts.reshape(-1, pts.shape[-1])
        # print(self.concat_plane)
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features,levels=level, num_levels=None,concat_planes=self.concat_plane)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,scales: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps,scales)

        return features

    @property
    def get_grid_parameters(self):
        return self.grids.parameters()

    def convert_coarse_to_fine(self,old_plane):

        for fb,grids in enumerate(self.grids):
            coo_combs = list(itertools.combinations(
        range(4), 2))
            for idx,grid in enumerate(grids):
                print(coo_combs[idx])
                x,y = coo_combs[idx]
                (_,outdim,H,W) = grid.shape
            #1.生成新的点的坐标
                # grid_y, grid_x = torch.meshgrid(
                # torch.linspace(-1, 1, H),
                # torch.linspace(-1, 1, W)
                # )
                # grid_y = grid_y.reshape(-1,H*W)
                # grid_x = grid_x.reshape(-1,H*W)
                
                x_max,x_min = self.aabb[0][x],self.aabb[1][x]
                x_max = (x_max-old_plane.aabb[1][x])*2/(old_plane.aabb[0][x]-old_plane.aabb[1][x]) -1.0
                x_min = (x_min-old_plane.aabb[1][x])*2/(old_plane.aabb[0][x]-old_plane.aabb[1][x]) -1.0
                # x_max = (x_max-old_plane.aabb[0][x])*2/(old_plane.aabb[1][x]-old_plane.aabb[0][x]) -1.0
                # x_min = (x_min-old_plane.aabb[0][x])*2/(old_plane.aabb[1][x]-old_plane.aabb[0][x]) -1.0
                if 3 in coo_combs[idx]:
                    y_max,y_min = 1,-1
                else:
                    y_max,y_min = self.aabb[0][y],self.aabb[1][y]
                    y_max = (y_max-old_plane.aabb[1][y])*2/(old_plane.aabb[0][y]-old_plane.aabb[1][y]) -1.0
                    y_min = (y_min-old_plane.aabb[1][y])*2/(old_plane.aabb[0][y]-old_plane.aabb[1][y]) -1.0

                    # y_max = (y_max-old_plane.aabb[0][y])*2/(old_plane.aabb[1][y]-old_plane.aabb[0][y]) -1.0
                    # y_min = (y_min-old_plane.aabb[0][y])*2/(old_plane.aabb[1][y]-old_plane.aabb[0][y]) -1.0

                grid_y,grid_x = torch.meshgrid(
                    torch.linspace(y_min,y_max,H),
                    torch.linspace(x_min,x_max,W)
                )
                grid = torch.cat((grid_x.unsqueeze(-1),grid_y.unsqueeze(-1)),dim=-1).unsqueeze(0).to("cuda")
                print(grid.shape)
                features = F.grid_sample(old_plane.grids[fb][idx],grid,mode='nearest',align_corners=True)
                print(features.device)
                self.grids[fb][idx] = nn.Parameter(features)
    
    def planetv(self):
        spatial_planes = [0,1,3]
        total = 0
        for grids in self.grids:
            # for idx,grid in enumerate(grids):
            for spatial_plane in spatial_planes:
                total += compute_plane_tv(grids[spatial_plane])
        return total
    
    def timesmooth(self):
        total = 0
        time_planes = [1,4,5]
        for grids in self.grids:
            for time_idx in time_planes:
                total += compute_plane_smoothness(grids[time_idx])
        return total