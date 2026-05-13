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
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, eval_sh
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.ir_utils import linear_to_srgb, sample_diffuse_directions, dot
import nvdiffrast.torch as dr
import envlight
from einops import rearrange

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

        self.metallic_activation = torch.sigmoid
        self.roughness_activation = torch.sigmoid


    def __init__(self, sh_degree : int, environment_texture=None, environment_scale=1.0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._albedo_init = torch.empty(0)
        self._metallic_init = torch.empty(0)
        self._roughness_init = torch.empty(0)
        self.diffuse_occ = torch.empty(0)
        self.FG_LUT = torch.from_numpy(
            np.fromfile("load/lights/bsdf_256_256.bin", dtype=np.float32).reshape(
                1, 256, 256, 2
            )).cuda()
        self.grid = torch.empty(0)
        self.min_pts = torch.empty(0)
        self.max_pts = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.diffuse_sample_num = 128
        self.specular_sample_num = 24
        self.envlight = envlight.EnvLight(environment_texture, scale=environment_scale, min_res=16, max_res=512, min_roughness=0.08, max_roughness=0.5, trainable=True).cuda()

        az, el = self.sample_sphere(self.diffuse_sample_num, 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi # scale to [0,1]
        self.diffuse_direction_samples = np.stack([az, el], -1)
        self.diffuse_direction_samples = torch.from_numpy(self.diffuse_direction_samples.astype(np.float32)).cuda() # [dn0,2]


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._albedo_init,
            self._metallic_init,
            self._roughness_init,
            self.diffuse_occ,
            self.grid,
            self.max_pts,
            self.min_pts,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.envlight.state_dict()
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._albedo_init,
        self._metallic_init,
        self._roughness_init,
        self.diffuse_occ,
        self.grid,
        self.max_pts,
        self.min_pts,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        envlight_state_dict) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.envlight.load_state_dict(envlight_state_dict)

    @property
    def get_albedo_init(self):
        return (self._albedo_init + 0.5).clamp(0.0,1.0)
    
    @property
    def get_metallic_init(self):
        return self._metallic_init.clamp(0.0, 1.0)

    @property
    def get_roughness_init(self):
        return self._roughness_init.clamp(0.0, 1.0)

    @property
    def get_rotation_matrix(self):
        return build_rotation(self._rotation)
    
    @property
    def get_FG_LUT(self):
        return self.FG_LUT
    
    @property
    def get_envlight(self):
        return self.envlight

    @property
    def get_eigenvector(self):
        scales = self.get_scaling
        N = scales.shape[0]
        idx = torch.min(scales, dim=1)[1]
        normals = self.get_rotation_matrix[np.arange(N), :, idx]
        normals = torch.nn.functional.normalize(normals, dim=1)
        return normals
    
    def sample_sphere(self, num_samples,begin_elevation = 0):
        """ sample angles from the sphere
        reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
        """
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.
        azimuths = []
        elevations = []
        for n in range(num_points - num_samples, num_points):
            z = 2. * n / num_points - 1.
            azimuths.append(2 * np.pi * n * phi % (2 * np.pi))
            elevations.append(np.arcsin(z))
        return np.array(azimuths), np.array(elevations)
    

    def get_sphere_intersection(self, pts, dirs, R=1):
        dtx = torch.sum(pts*dirs,dim=-1,keepdim=True) # rn,1
        xtx = torch.sum(pts**2,dim=-1,keepdim=True) # rn,1
        dist = dtx ** 2 - xtx + R**2
        assert torch.sum(dist<0)==0
        dist = -dtx + torch.sqrt(dist+1e-6) # rn,1
        return dist

    def query_pts_value(self, points, grid_num=128):
        normalized_pc = (points - self.min_pts) / (self.max_pts - self.min_pts)
        indices = (normalized_pc * (grid_num - 1)).long().clamp(0, 127)
        result = self.grid[indices[:, :, 0], indices[:, : , 1], indices[:, : , 2]]
        return result

    def compute_occlusion(self, n_dot_v, reflective, index=None):
        with torch.no_grad():
            if len(n_dot_v.shape)==3:
                xyz = self.get_xyz[index,:]
                P = xyz.shape[0]
                sn = reflective.shape[1]
                pts_occ_weight = torch.zeros(P,sn, device="cuda")
                distances = torch.norm(xyz, dim=1)
                max_radiu = torch.max(distances) + 0.25
                normal_mask = n_dot_v.squeeze() > 0.3
                sam_pts_num = 128
                if torch.sum(normal_mask)>0:
                    pts_filter = xyz[normal_mask].unsqueeze(1).repeat(1,sn,1).reshape(-1,3)
                    dirs = reflective[normal_mask].reshape(-1,3)
                    max_dist = self.get_sphere_intersection(pts_filter, dirs, max_radiu) # pn,1
                    z_vals = torch.linspace(0, 1, sam_pts_num)[16:].cuda()
                    z_vals = max_dist * z_vals.unsqueeze(0)
                    for i in range(128-16):
                        pts_sample = z_vals[:,i:i+1].unsqueeze(-1) * dirs.unsqueeze(-2) + pts_filter.unsqueeze(-2)
                        pts_weight = self.query_pts_value(pts_sample).squeeze(-1)
                        pts_occ_weight[normal_mask] += pts_weight.float().reshape(-1,sn)
                
                occ = pts_occ_weight.float().unsqueeze(-1).reshape(P,sn,1)
                occ = (occ > 1).float()
                return occ
            xyz = self.get_xyz
            P = xyz.shape[0]
            pts_occ_weight = torch.zeros(P, device="cuda")
            distances = torch.norm(xyz, dim=1)
            max_radiu = torch.max(distances) + 0.25
            normal_mask = n_dot_v.squeeze(-1) > 0.3
            sam_pts_num = 128
            if torch.sum(normal_mask)>0:
                pts_filter = xyz[normal_mask]
                dirs = reflective[normal_mask]
                N_filtered = pts_filter.shape[0]
                chunk_size = 4096
                pts_weight_all = torch.zeros(N_filtered, device="cuda")
                for cstart in range(0, N_filtered, chunk_size):
                    cend = min(cstart + chunk_size, N_filtered)
                    pf = pts_filter[cstart:cend]
                    dr = dirs[cstart:cend]
                    max_dist = self.get_sphere_intersection(pf, dr, max_radiu)
                    z_vals = torch.linspace(0, 1, sam_pts_num)[16:].cuda()
                    z_vals = max_dist * z_vals.unsqueeze(0)
                    pts_sample = z_vals.unsqueeze(-1) * dr.unsqueeze(-2) + pf.unsqueeze(-2)
                    pw = self.query_pts_value(pts_sample)
                    pts_weight_all[cstart:cend] = torch.sum(pw, dim=-1)
                pts_occ_weight[normal_mask] = pts_weight_all.float()
            
            occ = pts_occ_weight.float().unsqueeze(-1)
            occ = (occ > 1).float()
            return occ

    def get_grid(self, grid_num=128):
        with torch.no_grad():
            voxel_grid = torch.zeros((grid_num, grid_num, grid_num), dtype=torch.uint8, device="cuda")
            xyz = self.get_xyz
            min_vals, _ = torch.min(xyz, dim=0) 
            min_vals = min_vals - 0.1
            max_vals, _ = torch.max(xyz, dim=0)
            max_vals = max_vals + 0.1
            xyz = (xyz - min_vals) / (max_vals - min_vals)
            P = xyz.shape[0]
            scales = self.get_scaling
            rotations = self.get_rotation
            L = build_scaling_rotation(scales, rotations)
            covariance = L @ L.transpose(1, 2) * 3
            x111 = ((xyz + covariance[..., 0] + covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
            x110 = ((xyz + covariance[..., 0] + covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
            x101 = ((xyz + covariance[..., 0] - covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
            x100 = ((xyz + covariance[..., 0] - covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
            x011 = ((xyz - covariance[..., 0] + covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
            x010 = ((xyz - covariance[..., 0] + covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
            x001 = ((xyz - covariance[..., 0] - covariance[..., 1] + covariance[..., 2]) * (grid_num - 1)).long()
            x000 = ((xyz - covariance[..., 0] - covariance[..., 1] - covariance[..., 2]) * (grid_num - 1)).long()
            xyz = (xyz * (grid_num - 1)).long()
            indices = torch.cat([x111,x110,x101,x100,x011,x010,x001,x000,xyz], 0).clamp(0,127)
            voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            self.grid = voxel_grid
            self.min_pts = min_vals
            self.max_pts = max_vals

    def get_diffuse_occ(self):
        with torch.no_grad():
            self.get_grid()
            normal = self.get_eigenvector
            self.diffuse_occ = torch.zeros((self.get_xyz.shape[0], self.diffuse_sample_num), device=normal.device)
            diffuse_directions = sample_diffuse_directions(normal, self.diffuse_direction_samples, is_train=False)
            xyz = self.get_xyz
            distances = torch.norm(xyz, dim=1)
            max_radiu = torch.max(distances) + 0.25
            sam_pts_num = 128
            chunk_size = 4096  # Process points in chunks to avoid OOM
            P = xyz.shape[0]
            for i in range(self.diffuse_sample_num):
                dirs = diffuse_directions[:,i,:]
                for start in range(0, P, chunk_size):
                    end = min(start + chunk_size, P)
                    xyz_chunk = xyz[start:end]
                    dirs_chunk = dirs[start:end]
                    max_dist = self.get_sphere_intersection(xyz_chunk, dirs_chunk, max_radiu)
                    z_vals = torch.linspace(0, 1, sam_pts_num)[16:].cuda()
                    z_vals = max_dist * z_vals.unsqueeze(0)
                    pts_sample = z_vals.unsqueeze(-1) * dirs_chunk.unsqueeze(-2) + xyz_chunk.unsqueeze(-2)
                    pts_weight = self.query_pts_value(pts_sample)
                    pts_weight = torch.sum(pts_weight, dim=-1)
                    self.diffuse_occ[start:end, i] = (pts_weight > 1).float()

    def compute_color(self, camera_center, iteration=None, is_train=None, first_stage_step=5000, second_stage_step=30000, remove_noise=False, hdr_rotation=False, exposure=0.0):
        means3D = self.get_xyz
        if remove_noise:
            v = camera_center - means3D
            dis = torch.sum(v * v, dim=-1)
            xyz_mask = torch.where(dis > 1.25, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
            self._xyz.data = self._xyz.data * xyz_mask.unsqueeze(-1)
            means3D = self.get_xyz

        albedo = self.get_albedo_init
        shading_normal = self.get_eigenvector
        metallic = self.get_metallic_init
        roughness = self.get_roughness_init
        envlight = self.get_envlight

        v = camera_center - means3D
        v = torch.nn.functional.normalize(v, dim=1)
        n_dot_v = dot(v, shading_normal)
        reflective = n_dot_v * shading_normal * 2 - v

        occ = None
        if iteration > second_stage_step:
            shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, reflective)
            indirect_light = torch.clamp_min(sh2rgb + 0.5, 0.0)
            occ = self.compute_occlusion(n_dot_v, reflective)

        if hdr_rotation:
            reflective_x = reflective[:,0].unsqueeze(-1)
            reflective_y = reflective[:,1].unsqueeze(-1)
            reflective_z = reflective[:,2].unsqueeze(-1)

            reflective = torch.cat([-reflective_y, reflective_z, -reflective_x], dim=-1) #rot_y(90)@rot_x(-90)

        prefix_shape = albedo.shape[:-1]
        diffuse_albedo = (1 - metallic) * albedo
        fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)
        # --- THE ULTIMATE PYTORCH BYPASS ---
        import torch.nn.functional as F
        lut = self.get_FG_LUT.permute(0, 3, 1, 2) # Convert to PyTorch layout [1, 2, 256, 256]
        
        # Convert OpenGL (0 to 1) coordinates to PyTorch (-1 to 1) coordinates
        u = fg_uv[:, 0] * 2.0 - 1.0
        v = 1.0 - 2.0 * fg_uv[:, 1]
        grid = torch.stack([u, v], dim=-1).reshape(1, -1, 1, 2) 
        
        # Sample the texture natively
        fg = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border", align_corners=False)
        fg = fg.permute(0, 2, 3, 1).reshape(*prefix_shape, 2)
        # -----------------------------------
        F0 = (1 - metallic) * 0.04 + metallic * albedo
        specular_albedo = F0 * fg[:, 0:1] + fg[:, 1:2]
        if is_train:
            envlight.build_base()
        envlight.build_mips()
        if iteration > second_stage_step:
            # Process diffuse lighting in point-chunks to avoid OOM
            _pchunk = 4096
            P = shading_normal.shape[0]
            diffuse_light = torch.zeros(P, 3, device="cuda")
            for pstart in range(0, P, _pchunk):
                pend = min(pstart + _pchunk, P)
                chunk_dirs = sample_diffuse_directions(shading_normal[pstart:pend], self.diffuse_direction_samples, is_train=False)
                chunk_dirs = rearrange(chunk_dirs, "B N C -> (B N) C")
                if hdr_rotation:
                    dx = chunk_dirs[:,0].unsqueeze(-1)
                    dy = chunk_dirs[:,1].unsqueeze(-1)
                    dz = chunk_dirs[:,2].unsqueeze(-1)
                    chunk_dirs = torch.cat([-dy, dz, -dx], dim=-1)
                chunk_light = envlight(chunk_dirs)
                chunk_occ = rearrange(self.diffuse_occ[pstart:pend], "B N -> (B N)").unsqueeze(1)
                chunk_light = (1 - chunk_occ) * chunk_light
                chunk_light = rearrange(chunk_light, "(B N) C -> B N C", N=self.diffuse_sample_num)
                diffuse_light[pstart:pend] = torch.mean(chunk_light, dim=1)
        else:
            if hdr_rotation:
                normal_x = shading_normal[:,0].unsqueeze(-1)
                normal_y = shading_normal[:,1].unsqueeze(-1)
                normal_z = shading_normal[:,2].unsqueeze(-1)
                shading_normal = torch.cat([-normal_y, normal_z, -normal_x], dim=-1)
            diffuse_light = envlight(shading_normal)
        diffuse_color = diffuse_albedo * diffuse_light
        specular_light = envlight(reflective, roughness)
        if occ is not None:
            specular_indirect_light = occ * indirect_light
            specular_direct_light = (1-occ) * specular_light
            specular_indirect_color = specular_indirect_light * specular_albedo
            specular_direct_color = specular_direct_light * specular_albedo
            specular_light = specular_indirect_light + specular_direct_light
            specular_color = specular_indirect_color + specular_direct_color
        else:
            specular_color = specular_light * specular_albedo

        color = linear_to_srgb((diffuse_color + specular_color)*2**exposure).clamp(0.0, 1.0)
        mask = torch.where(n_dot_v > 0, torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda"))
        mask2 = (torch.rand_like(mask) < 0.3).float()
        if is_train and iteration > (first_stage_step+5000):
                color = color * mask + torch.rand_like(color) * (1 - mask) * mask2
        else:
            color = color * mask

        result = {
            "color": color,
            "albedo": linear_to_srgb(albedo).clamp(0.0, 1.0),
            "diffuse_albedo": linear_to_srgb(diffuse_albedo).clamp(0.0, 1.0),
            "diffuse_light": linear_to_srgb(diffuse_light).clamp(0.0, 1.0),
            "diffuse_color": linear_to_srgb(diffuse_color*2**exposure).clamp(0.0, 1.0),
            "specular_albedo": linear_to_srgb(specular_albedo).clamp(0.0, 1.0),
            "specular_light": linear_to_srgb(specular_light).clamp(0.0, 1.0),
            "specular_color": linear_to_srgb(specular_color*2**exposure).clamp(0.0, 1.0),
            "specular_indirect_light": None,
            "specular_direct_light": None,
            "specular_indirect_color": None,
            "specular_direct_color": None,
            "occ": None
        }
        if occ is not None:
            result.update({
            "specular_indirect_light": linear_to_srgb(specular_indirect_light).clamp(0.0, 1.0),
            "specular_direct_light": linear_to_srgb(specular_direct_light).clamp(0.0, 1.0),
            "specular_indirect_color": linear_to_srgb(specular_indirect_color).clamp(0.0, 1.0),
            "specular_direct_color": linear_to_srgb(specular_direct_color).clamp(0.0, 1.0),
            "occ": occ
            })
        return result

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
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

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.zeros(fused_point_cloud.shape[0], 3).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        metallic = torch.rand(fused_point_cloud.shape[0], 1).float().cuda()
        roughness = torch.rand(fused_point_cloud.shape[0], 1).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        num_points = fused_point_cloud.shape[0]
        dist2 = torch.full((num_points,), 0.01, dtype=torch.float32, device="cuda")
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.diffuse_occ = torch.ones((fused_point_cloud.shape[0], self.diffuse_sample_num), device="cuda")

        self._albedo_init = nn.Parameter(fused_color.requires_grad_(True))
        self._metallic_init = nn.Parameter(metallic.requires_grad_(True))
        self._roughness_init = nn.Parameter(roughness.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

            {'params': [self._albedo_init], 'lr': training_args.albedo_lr_init, "name": "albedo_init"},
            {'params': [self._metallic_init], 'lr': training_args.material_lr_init, "name": "metallic_init"},
            {'params': [self._roughness_init], 'lr': training_args.material_lr_init, "name": "roughness_init"},
            {'params': self.envlight.net.parameters(), 'lr': training_args.hdr_lr_init, "name": "hdr_net"},
            {'params': [self.envlight.init_base], 'lr': training_args.hdr_lr_init, "name": "hdr_init_base"},
            {'params': [self.envlight.base_train], 'lr': training_args.hdr_base_lr_init, "name": "hdr_base_train"},

        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.hdr_scheduler_args = get_expon_lr_func(lr_init=training_args.hdr_lr_init,
                                                    lr_final=training_args.hdr_lr_final,
                                                    lr_delay_mult=training_args.hdr_lr_delay_mult,
                                                    max_steps=training_args.hdr_lr_max_steps)
        self.albedo_scheduler_args = get_expon_lr_func(lr_init=training_args.albedo_lr_init,
                                                    lr_final=training_args.albedo_lr_final,
                                                    lr_delay_mult=training_args.albedo_lr_delay_mult,
                                                    max_steps=training_args.albedo_lr_max_steps)
        self.material_scheduler_args = get_expon_lr_func(lr_init=training_args.material_lr_init,
                                                    lr_final=training_args.material_lr_final,
                                                    lr_delay_mult=training_args.material_lr_delay_mult,
                                                    max_steps=training_args.material_lr_max_steps)
        self.hdr_base_scheduler_args = get_expon_lr_func(lr_init=training_args.hdr_base_lr_init,
                                                    lr_final=training_args.hdr_base_lr_final,
                                                    lr_delay_mult=training_args.hdr_base_lr_delay_mult,
                                                    max_steps=training_args.hdr_base_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "hdr_net":
                lr = self.hdr_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "hdr_init_base":
                lr = self.hdr_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "hdr_base_train":
                lr = self.hdr_base_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "metallic_init":
                lr = self.material_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "roughness_init":
                lr = self.material_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr
            if param_group["name"] == "albedo_init":
                lr = self.albedo_scheduler_args(max(0, iteration-5000))
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('albedo_r')
        l.append('albedo_g')
        l.append('albedo_b')
        l.append('metallic')
        l.append('roughness')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        #normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        albedo_init = self._albedo_init.detach().cpu().numpy()
        metallic_init = self._metallic_init.detach().cpu().numpy()
        roughness_init = self._roughness_init.detach().cpu().numpy()
        normals = self.get_eigenvector.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, albedo_init, metallic_init, roughness_init), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        albedo_init = np.stack((np.asarray(plydata.elements[0]["albedo_r"]),
                        np.asarray(plydata.elements[0]["albedo_g"]),
                        np.asarray(plydata.elements[0]["albedo_b"])),  axis=1)
        metallic_init = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
        roughness_init = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._albedo_init = nn.Parameter(torch.tensor(albedo_init, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic_init = nn.Parameter(torch.tensor(metallic_init, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness_init = nn.Parameter(torch.tensor(roughness_init, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
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
            if "hdr" in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
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

        self._albedo_init = optimizable_tensors["albedo_init"]
        self._metallic_init = optimizable_tensors["metallic_init"]
        self._roughness_init = optimizable_tensors["roughness_init"]
        self.diffuse_occ = self.diffuse_occ[valid_points_mask]


        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "hdr" in group["name"]:
                continue
            assert len(group["params"]) == 1
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_albedo_init, new_metallic_init, new_roughness_init, new_diffuse_occ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "albedo_init": new_albedo_init,
        "metallic_init": new_metallic_init,
        "roughness_init": new_roughness_init,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._albedo_init = optimizable_tensors["albedo_init"]
        self._metallic_init = optimizable_tensors["metallic_init"]
        self._roughness_init = optimizable_tensors["roughness_init"]
        self.diffuse_occ = torch.cat([self.diffuse_occ, new_diffuse_occ], 0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
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
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_albedo_init = self._albedo_init[selected_pts_mask].repeat(N,1)
        new_metallic_init = self._metallic_init[selected_pts_mask].repeat(N,1)
        new_roughness_init = self._roughness_init[selected_pts_mask].repeat(N,1)
        new_diffuse_occ = self.diffuse_occ[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_albedo_init, new_metallic_init, new_roughness_init, new_diffuse_occ)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_albedo_init = self._albedo_init[selected_pts_mask]
        new_metallic_init = self._metallic_init[selected_pts_mask]
        new_roughness_init = self._roughness_init[selected_pts_mask]
        new_diffuse_occ = self.diffuse_occ[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_albedo_init, new_metallic_init, new_roughness_init, new_diffuse_occ)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1