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
from scene.gaussian_model import BasicPointCloud

# Sentinel value for a *_gt_dir: instead of loading prior images from disk, use a
# constant 0 for every pixel (e.g. metallic_gt_dir="metallic_simulated_zero"
# means the scene contains no metallic objects, so the metallic GT is all zeros).
METALLIC_SIMULATED_ZERO = "metallic_simulated_zero"

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    exposure: float
    albedo_gt: object = None   # PIL Image or None
    normal_gt: object = None   # PIL Image or None
    metallic_gt: float = None  # scalar (e.g. 0.0) or None
    roughness_gt: float = None # scalar (e.g. 0.5) or None
    normal_in_camera_space: bool = False  # True for real-world (COLMAP) priors

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

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

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, exposure=0.0)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
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

def readColmapSceneInfo(path, images, eval, llffhold=8):
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
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", gt_priors_dir=None, albedo_dir="albedo_gt", normal_dir="normal_gt", metallic_dir="", roughness_dir=""):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        if "camera_angle_x" not in contents.keys():
            fovx = None
        else:
            fovx = contents["camera_angle_x"] 

        if "exposure" in contents.keys():
            exposure = frame["exposure"]
        else:
            exposure = 0.0


        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            arr = np.concatenate((arr, norm_data[...,3:4]),-1)
            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGBA")

            if fovx == None:
                focal_length = contents["fl_x"]
                FovY = focal2fov(focal_length, image.size[1])
                FovX = focal2fov(focal_length, image.size[0])
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovx 
                FovX = fovy

            # --- Load GT priors from the configured per-property folders ---
            # Each prior reads <gt_priors_dir>/<folder>/<prop>_<idx>.png; the file
            # prefix is the property name (albedo/normal/metallic/roughness)
            # regardless of which folder variant (e.g. *_gt, *_video) is chosen.
            # A folder set to "" disables that prior (left as None).
            albedo_gt_img = None
            normal_gt_img = None
            metallic_gt = None
            roughness_gt = None

            if gt_priors_dir is not None:
                # Derive frame index from file_path, e.g. "./train/rgba/rgba_042" -> "042"
                basename = os.path.basename(frame["file_path"])  # "rgba_042"
                frame_idx_str = basename.split("_")[-1]  # "042"

                def _load_prior(folder, prop):
                    if not folder:
                        return None
                    prior_path = os.path.join(gt_priors_dir, folder, f"{prop}_{frame_idx_str}.png")
                    if os.path.exists(prior_path):
                        return Image.open(prior_path)
                    print(f"[WARNING] {prop} prior not found: {prior_path}")
                    return None

                albedo_gt_img = _load_prior(albedo_dir, "albedo")
                normal_gt_img = _load_prior(normal_dir, "normal")
                if metallic_dir == METALLIC_SIMULATED_ZERO:
                    metallic_gt = 0.0  # no metallic objects: all-zero GT (no disk load)
                else:
                    metallic_gt = _load_prior(metallic_dir, "metallic")
                roughness_gt = _load_prior(roughness_dir, "roughness")

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], exposure=exposure,
                            albedo_gt=albedo_gt_img, normal_gt=normal_gt_img, metallic_gt=metallic_gt, roughness_gt=roughness_gt))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
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
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def isSyntheticWithPriors(path):
    """Detect whether a dataset path is a synthetic-with-priors dataset.

    Synthetic-with-priors datasets (e.g. armadillo, lego) are distinguished
    from vanilla Blender datasets by having a train/rgba/ subdirectory
    alongside transforms_train.json.
    """
    has_transforms = os.path.exists(os.path.join(path, "transforms_train.json"))
    has_rgba_subdir = os.path.isdir(os.path.join(path, "train", "rgba"))
    return has_transforms and has_rgba_subdir

def readSyntheticWithPriorsInfo(path, white_background, eval, extension=".png",
                                albedo_dir="albedo_gt", normal_dir="normal_gt",
                                metallic_dir="", roughness_dir=""):
    """Read a synthetic dataset from the datasets_with_priors directory.

    These datasets (e.g. armadillo, lego) use the Blender/synthetic camera
    convention with transforms_*.json files, and store images under
    {train,val,test}/rgba/.  Each GT prior (albedo/normal/metallic/roughness)
    is read from a configurable per-property folder (e.g. albedo_gt,
    albedo_video, albedo); a folder set to "" disables that prior.
    """
    train_gt_dir = os.path.join(path, "train")
    test_gt_dir = os.path.join(path, "test")

    configured_dirs = [d for d in (albedo_dir, normal_dir, metallic_dir, roughness_dir) if d]

    def _found_dirs(split_dir):
        return [d for d in configured_dirs if os.path.isdir(os.path.join(split_dir, d))]

    train_found = _found_dirs(train_gt_dir)
    test_found = _found_dirs(test_gt_dir)
    has_train_gt = len(train_found) > 0
    has_test_gt = len(test_found) > 0

    if has_train_gt:
        print(f"Found GT prior folders in train split: {train_found}")
    if has_test_gt:
        print(f"Found GT prior folders in test split: {test_found}")

    print("Reading Synthetic-with-Priors Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension,
        gt_priors_dir=train_gt_dir if has_train_gt else None,
        albedo_dir=albedo_dir, normal_dir=normal_dir,
        metallic_dir=metallic_dir, roughness_dir=roughness_dir)

    print("Reading Synthetic-with-Priors Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension,
        gt_priors_dir=test_gt_dir if has_test_gt else None,
        albedo_dir=albedo_dir, normal_dir=normal_dir,
        metallic_dir=metallic_dir, roughness_dir=roughness_dir)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Synthetic datasets have no COLMAP data, so we initialise with
        # random points inside the scene bounds.
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def isColmapWithPriors(path):
    """Detect a real-world COLMAP dataset that ships diffusion priors.

    These datasets (e.g. bicycle, garden) have all frames directly under a
    rgba/ folder (no train/val/test split), a COLMAP `sparse/` reconstruction
    for the camera poses, and per-property prior folders (albedo/, normal/, ...)
    at the dataset root.  They are distinguished from the synthetic-with-priors
    datasets by the ABSENCE of transforms_train.json.
    """
    has_sparse = os.path.isdir(os.path.join(path, "sparse"))
    has_rgba = os.path.isdir(os.path.join(path, "rgba"))
    has_transforms = os.path.exists(os.path.join(path, "transforms_train.json"))
    return has_sparse and has_rgba and not has_transforms


def _colmap_rgba_lookup(rgba_dir):
    """Map a sequential frame index -> rgba file path.

    The prior-extraction pipeline renames the COLMAP images to
    <dataset>_<idx:03d>.<ext> in sorted-name order, so the i-th camera (after
    sorting the COLMAP extrinsics by name) corresponds to <prefix>_<i:03d>.
    Returns (prefix, ext, {idx: filepath}).
    """
    files = sorted(os.listdir(rgba_dir))
    mapping = {}
    prefix, ext = None, None
    for f in files:
        stem, e = os.path.splitext(f)
        if "_" not in stem:
            continue
        pfx, idx_str = stem.rsplit("_", 1)
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)
        mapping[idx] = os.path.join(rgba_dir, f)
        prefix, ext = pfx, e
    return prefix, ext, mapping


def readColmapWithPriorsInfo(path, eval, llffhold=8,
                             albedo_dir="albedo", normal_dir="normal",
                             metallic_dir="", roughness_dir=""):
    """Read a real-world COLMAP dataset that ships diffusion priors.

    Camera poses come from the COLMAP `sparse/0` reconstruction; RGB frames live
    in rgba/ and the GT-style priors in per-property folders at the dataset root
    (albedo/, normal/, ...). The i-th camera (COLMAP extrinsics sorted by name)
    maps to rgba/<prefix>_<i:03d>.<ext> and <prop_dir>/<prop>_<i:03d>.png, which
    matches the renaming done by the prior-extraction pipeline.

    The normal priors are stored in CAMERA space (they come from a per-view
    diffusion model), so each CameraInfo is flagged `normal_in_camera_space=True`
    and the training code rotates them into world space using the camera pose.
    """
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(path, "sparse/0", "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(path, "sparse/0", "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(path, "sparse/0", "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "sparse/0", "cameras.txt"))

    rgba_dir = os.path.join(path, "rgba")
    prefix, ext, rgba_map = _colmap_rgba_lookup(rgba_dir)

    # Sort the COLMAP cameras by name so the sequential index matches the
    # <prefix>_<idx> renaming performed during prior extraction.
    sorted_keys = sorted(cam_extrinsics, key=lambda k: cam_extrinsics[k].name)

    def _load_prior(folder, prop, frame_idx):
        if not folder:
            return None
        prior_path = os.path.join(path, folder, f"{prop}_{frame_idx:03d}.png")
        if os.path.exists(prior_path):
            return Image.open(prior_path)
        print(f"[WARNING] {prop} prior not found: {prior_path}")
        return None

    configured = [d for d in (albedo_dir, normal_dir, metallic_dir, roughness_dir) if d]
    found = [d for d in configured if os.path.isdir(os.path.join(path, d))]
    if found:
        print(f"Found COLMAP prior folders: {found}")

    cam_infos = []
    for idx, key in enumerate(sorted_keys):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        R = np.transpose(qvec2rotmat(extr.qvec))  # camera-to-world rotation
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = rgba_map.get(idx)
        if image_path is None or not os.path.exists(image_path):
            print(f"[WARNING] rgba frame not found for index {idx} "
                  f"(expected {prefix}_{idx:03d}{ext}); skipping camera.")
            continue
        image_name = Path(image_path).stem
        # Real photos have no alpha; force RGBA so the (all-opaque) mask exists.
        image = Image.open(image_path).convert("RGBA")

        albedo_gt_img = _load_prior(albedo_dir, "albedo", idx)
        normal_gt_img = _load_prior(normal_dir, "normal", idx)
        if metallic_dir == METALLIC_SIMULATED_ZERO:
            metallic_gt = 0.0  # no metallic objects: all-zero GT (no disk load)
        else:
            metallic_gt = _load_prior(metallic_dir, "metallic", idx)
        roughness_gt = _load_prior(roughness_dir, "roughness", idx)

        cam_infos.append(CameraInfo(
            uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            image_path=image_path, image_name=image_name,
            width=image.size[0], height=image.size[1], exposure=0.0,
            albedo_gt=albedo_gt_img, normal_gt=normal_gt_img,
            metallic_gt=metallic_gt, roughness_gt=roughness_gt,
            normal_in_camera_space=True))

    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for i, c in enumerate(cam_infos) if i % llffhold != 0]
        test_cam_infos = [c for i, c in enumerate(cam_infos) if i % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "SyntheticWithPriors": readSyntheticWithPriorsInfo,
    "ColmapWithPriors": readColmapWithPriorsInfo,
}

