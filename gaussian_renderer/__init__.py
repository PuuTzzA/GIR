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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import RGB2SH

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, random_bg_color = None, iteration=None, scaling_modifier = 1.0, is_train=None, first_stage_step=5000, second_stage_step=30000, remove_noise=False, hdr_rotation=False, albedo_geometry_warmup=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if random_bg_color is not None:
        random_raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=random_bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        random_rasterizer = GaussianRasterizer(raster_settings=random_raster_settings)
    else:
        random_rasterizer = rasterizer

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    diffuse_color = None
    specular_indirect_light = None
    specular_indirect_color = None
    # In albedo geometry-warmup mode the flat per-gaussian albedo is rendered
    # (and supervised against the GT albedo prior) through BOTH the geometry and
    # the normal-alignment stages, i.e. for every iteration up to
    # second_stage_step, rather than only the first stage. The warmed-up albedo
    # is then carried into the PBR stage (its boundary reset is skipped below).
    use_flat_albedo = (iteration <= first_stage_step) or (albedo_geometry_warmup and iteration <= second_stage_step)
    if use_flat_albedo:
        colors_precomp = pc.get_albedo_init
        albedo = pc.get_albedo_init  # alias so the Phase-2 viz/logging path works
        # Original GIR zeroes the warm-up albedo at the first-stage boundary so
        # the PBR albedo starts fresh; in warm-up mode we keep it. Boundary
        # resets are a training-time state transition, so only fire on is_train.
        if iteration == first_stage_step and is_train and not albedo_geometry_warmup:
            pc._albedo_init.data = torch.zeros_like(pc._albedo_init)
    else:
        if iteration == second_stage_step+1 and is_train:
            # Enter the PBR stage: reset the radiance features and randomise the
            # material channels. In warm-up mode keep the albedo learned during
            # the warm-up (do NOT zero _albedo_init).
            if not albedo_geometry_warmup:
                pc._albedo_init.data = torch.zeros_like(pc._albedo_init)
            pc._features_dc.data = torch.zeros_like(pc._features_dc)
            pc._features_rest.data = torch.zeros_like(pc._features_rest)
            pc._metallic_init.data = torch.rand_like(pc._metallic_init) * 0.2
            pc._roughness_init.data = torch.rand_like(pc._roughness_init)
        result = pc.compute_color(viewpoint_camera.camera_center, iteration, is_train, first_stage_step, second_stage_step, remove_noise, hdr_rotation, viewpoint_camera.exposure)

        colors_precomp = result["color"]
        albedo = result["albedo"]
        diffuse_albedo = result["diffuse_albedo"]
        diffuse_light = result["diffuse_light"]
        diffuse_color = result["diffuse_color"]
        specular_albedo = result["specular_albedo"]
        specular_indirect_light = result["specular_indirect_light"]
        specular_direct_light = result["specular_direct_light"]
        specular_indirect_color = result["specular_indirect_color"]
        specular_direct_color = result["specular_direct_color"]
        specular_light = result["specular_light"]
        specular_color = result["specular_color"]
        occ = result["occ"]

    rendered_image, radii, depth, alpha = random_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_normal = None
    rendered_metallic = None
    rendered_roughness = None
    rendered_albedo = None
    rendered_diffuse_color = None
    rendered_specular_color = None
    rendered_diffuse_albedo=None
    rendered_specular_albedo=None
    rendered_diffuse_light = None
    rendered_specular_light = None
    rendered_specular_indirect_color = None
    rendered_specular_indirect_light = None
    rendered_specular_direct_light = None
    rendered_specular_direct_color = None
    rendered_occ = None
    if iteration > second_stage_step:
        render_normal = (pc.get_eigenvector + 1) / 2
        render_material = torch.cat([pc.get_metallic_init, pc.get_roughness_init.clamp(0.08, 0.5), torch.zeros((render_normal.shape[0],1), device="cuda")], -1)
        rendered_normal, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = render_normal,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        rendered_material, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = render_material,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        rendered_metallic = rendered_material[0:1,...].repeat(3,1,1)
        rendered_roughness = rendered_material[1:2,...].repeat(3,1,1)
        rendered_albedo, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = albedo,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    elif iteration > first_stage_step:
        # Phase 2 (normal alignment): the geometry-derived shading normal must be
        # differentiable EVERY iteration so the GT-normal prior can supervise the
        # gaussian rotation / scaling here. Albedo & material are not optimized
        # yet in this phase, so they are rendered only periodically (no_grad) for
        # visualization / logging.
        render_normal = (pc.get_eigenvector + 1) / 2
        rendered_normal, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = render_normal,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        if iteration % 500 == 0:
            with torch.no_grad():
                render_material = torch.cat([pc.get_metallic_init, pc.get_roughness_init.clamp(0.08, 0.5), torch.zeros((render_normal.shape[0],1), device="cuda")], -1)
                rendered_material, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = render_material,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_metallic = rendered_material[0:1,...].repeat(3,1,1)
                rendered_roughness = rendered_material[1:2,...].repeat(3,1,1)
                rendered_albedo, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = albedo,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
    if iteration % 500 == 0:
        with torch.no_grad():
            if diffuse_color is not None:
                rendered_diffuse_color, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = diffuse_color,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_specular_color, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = specular_color,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_diffuse_albedo, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = diffuse_albedo,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_specular_albedo, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = specular_albedo,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_diffuse_light, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = diffuse_light,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_specular_light, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = specular_light,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
            if specular_indirect_color is not None:
                rendered_specular_indirect_color, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = specular_indirect_color,
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)
                rendered_specular_direct_color, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = specular_direct_color,
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)
            if specular_indirect_light is not None:
                rendered_specular_indirect_light, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = specular_indirect_light,
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)
                rendered_specular_direct_light, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = specular_direct_light,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_occ, _, _, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = occ.repeat(1,3),
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth,
            "alpha": alpha,
            "rendered_normal": rendered_normal,
            "rendered_albedo": rendered_albedo,
            "rendered_metallic": rendered_metallic,
            "rendered_roughness": rendered_roughness,
            "rendered_diffuse_color": rendered_diffuse_color,
            "rendered_specular_color": rendered_specular_color,
            "rendered_diffuse_light": rendered_diffuse_light,
            "rendered_specular_light": rendered_specular_light,
            "rendered_diffuse_albedo": rendered_diffuse_albedo,
            "rendered_specular_albedo": rendered_specular_albedo,
            "rendered_specular_indirect_light": rendered_specular_indirect_light,
            "rendered_specular_direct_light": rendered_specular_direct_light,
            "rendered_specular_indirect_color": rendered_specular_indirect_color,
            "rendered_specular_direct_color": rendered_specular_direct_color,
            "rendered_occ": rendered_occ,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
