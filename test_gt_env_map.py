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

import cv2
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import numpy as np
import sys
import importlib
from utils.sh_rotate_utils import Rotation
from utils.sh_vis_utils import shReconstructDiffuseMap, applyWindowing
from utils.normal_utils import compute_normal_world_space
import imageio.v2 as im
from skimage.metrics import structural_similarity as ssim_skimage
from utils.loss_utils import mse2psnr, img2mae, img2mse, img2mse_image
from utils.sh_vis_utils import getCoefficientsFromImage
import matplotlib.pyplot as plt

TINY_NUMBER = 1e-6


def process_environment_map_image(img_path, scale_high, threshold):
    
    img = plt.imread(img_path)
    img = torch.from_numpy(img).float() / 255
    img[img > threshold] *= scale_high
    coeffs = getCoefficientsFromImage(img.numpy(), 2)

    #apply peter-pike sloan windowing (optionally, to avoid negative signal)
    #coeffs = applyWindowing(coeffs=coeffs)
    return coeffs


def render_set(dataset : ModelParams, iteration : int, pipeline : PipelineParams, opt: OptimizationParams, test_config: str):

    dataset.eval = True
    gaussians = GaussianModel(dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a)
    scene = Scene(dataset, gaussians, load_iteration=iteration)

    if gaussians.with_mlp:
        gaussians.mlp.eval()
        gaussians.embedding.eval()

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # save dirs
    gt_path = os.path.join(dataset.model_path, "test_with_gt_env_map/gt_image")
    makedirs(gt_path, exist_ok=True)
    render_path = os.path.join(dataset.model_path, "test_with_gt_env_map/render_gt_env_map", "ours_{}".format(scene.loaded_iter))
    makedirs(render_path, exist_ok=True)

    # read test config
    sys.path.append(test_config)
    config = importlib.import_module("test_config").config
    
    #get test cameras for eval with envmap
    tmp_cameras = scene.getTestCameras()
    test_cameras = [c for c in tmp_cameras if c.image_name in config.keys()]

    ssims_unshadowed_scikit, psnrs_unshadowed, mse_unshadowed, mae_unshadowed = [], [], [], []
    ssims_shadowed_scikit, psnrs_shadowed, mse_shadowed, mae_shadowed = [],[],[],[]
    img_names, used_angles =[], []

    for viewpoint_cam in tqdm(test_cameras):
        print(viewpoint_cam.image_name)

        image_config = config[viewpoint_cam.image_name]
        mask_path = image_config["mask_path"]
        envmap_img_path = image_config["env_map_path"]
        init_rot_x = image_config["initial_env_map_rotation"]["x"]
        init_rot_y = image_config["initial_env_map_rotation"]["y"]
        init_rot_z = image_config["initial_env_map_rotation"]["z"]
        threshold = image_config["env_map_scaling"]["threshold"]
        scale = image_config["env_map_scaling"]["scale"]
        sun_angle_range = image_config["sun_angles"]

        
        # get processed env map
        env_sh_gt = process_environment_map_image(envmap_img_path, scale, threshold)

        #albedo
        rgb_precomp_alb =gaussians.get_albedo
        render_pkg = render(viewpoint_cam, gaussians, pipeline, background, override_color=rgb_precomp_alb)
        albedo = render_pkg["render"]
        
        # get gt
        gt_image = viewpoint_cam.original_image.cuda()
        torchvision.utils.save_image(gt_image, os.path.join(gt_path, viewpoint_cam.image_name))

        #get eval mask
        mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask=cv2.resize(mask, (gt_image.shape[2], gt_image.shape[1]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask=torch.from_numpy(mask//255).cuda()

        # get normals
        quaternions = gaussians.get_rotation
        scales = gaussians.get_scaling
        normal_vectors, multiplier = compute_normal_world_space(quaternions, scales, viewpoint_cam.world_view_transform, gaussians.get_xyz)
            
        best_psnr = 0
        best_angle = None
        
        n = 51
        sun_angles_prepare_list = torch.linspace(sun_angle_range[0], sun_angle_range[1], n)  
        sun_angles = [torch.tensor([angle,0, 0]) for angle in sun_angles_prepare_list] #rotate only around y
        
        for angle in tqdm(sun_angles):

            # first we adjust env map horizontally - match ground
            rotation = Rotation()
            rot = np.float32(np.dot(rotation.rot_y(init_rot_y), np.dot(rotation.rot_x(init_rot_x), rotation.rot_z(init_rot_z))))
            env_sh = np.matmul(rot, env_sh_gt)

            # then rotate around vertical axis y
            rotation = Rotation()
            rot = np.float32(np.dot(rotation.rot_y(angle[0]), np.dot(rotation.rot_x(angle[1]), rotation.rot_z(angle[2]))))
            env_sh = np.matmul(rot, env_sh)

            # get best psnr from manually set angle range - SOLNERF protocol
            env_sh_torch=torch.tensor(env_sh.T, dtype=torch.float32).cuda()
            rgb_precomp, _ = gaussians.compute_gaussian_rgb(env_sh_torch, multiplier = multiplier)
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background, override_color=rgb_precomp)
            rendering_shadowed = torch.clamp(render_pkg["render"], 0.0, 1.0)
            current_psnr = mse2psnr(img2mse(rendering_shadowed, gt_image, mask=mask))

            if current_psnr > best_psnr:
               best_angle = angle
               best_psnr = current_psnr
        
        # render all imgs for best angle
        rotation = Rotation()
        rot = np.float32(np.dot(rotation.rot_y(init_rot_y), np.dot(rotation.rot_x(init_rot_x), rotation.rot_z(init_rot_z))))
        env_sh = np.matmul(rot, env_sh_gt)

        rotation = Rotation()
        rot = np.float32(np.dot(rotation.rot_y(best_angle[0]), np.dot(rotation.rot_x(best_angle[1]), rotation.rot_z(best_angle[2]))))
        env_sh = np.matmul(rot, env_sh)
        env_sh_torch = torch.tensor(env_sh.T, dtype=torch.float32).cuda()
    
        #unshadowed version
        rgb_precomp_unshadowed, lum_unshadowed_precomp =gaussians.compute_gaussian_rgb(env_sh_torch, shadowed=False, normal_vectors=normal_vectors)
        rendering_unshadowed = render(viewpoint_cam, gaussians, pipeline, background, override_color=rgb_precomp_unshadowed)["render"]
        rendering_unshadowed = torch.clamp(rendering_unshadowed, 0.0, 1.0)
        
        illuminance_unshadowed = render(viewpoint_cam, gaussians, pipeline, background, override_color=lum_unshadowed_precomp)["render"]

        if illuminance_unshadowed.max()>1:
            illuminance_unshadowed /= illuminance_unshadowed.max()
        torchvision.utils.save_image(torch.clamp(illuminance_unshadowed*mask,0.0, 1.0), os.path.join(render_path, "unshadowed_luminance_"+viewpoint_cam.image_name))
        torchvision.utils.save_image(rendering_unshadowed*mask, os.path.join(render_path, "unshadowed_"+viewpoint_cam.image_name))

        # shadowed
        rgb_precomp, lum_shadowed_precomp = gaussians.compute_gaussian_rgb(env_sh_torch, multiplier = multiplier)
        render_pkg = render(viewpoint_cam, gaussians, pipeline, background, override_color=rgb_precomp)
        rendering_shadowed = torch.clamp(render_pkg["render"], 0.0, 1.0)

        illuminance_shadowed = render(viewpoint_cam, gaussians, pipeline, background, override_color=lum_shadowed_precomp)["render"]
        if illuminance_shadowed.max()>1:
            illuminance_shadowed /= illuminance_shadowed.max()
        torchvision.utils.save_image(torch.clamp(illuminance_shadowed*mask, 0.0, 1.0), os.path.join(render_path, "shadowed_luminance_"+viewpoint_cam.image_name))
        torchvision.utils.save_image(rendering_shadowed*mask, os.path.join(render_path, "shadowed_"+viewpoint_cam.image_name))
        

        used_angles.append(best_angle)
        img_names.append(viewpoint_cam.image_name)
        
        # Compute metrics
        psnrs_unshadowed.append(mse2psnr(img2mse(rendering_unshadowed, gt_image, mask=mask)))
        mae_unshadowed.append(img2mae(rendering_unshadowed, gt_image, mask=mask))
        mse_unshadowed.append(img2mse(rendering_unshadowed, gt_image, mask=mask))
        
        psnrs_shadowed.append(mse2psnr(img2mse(rendering_shadowed, gt_image, mask=mask)))
        mae_shadowed.append(img2mae(rendering_shadowed, gt_image, mask=mask))
        mse_shadowed.append(img2mse(rendering_shadowed, gt_image, mask=mask))

        unshadowed_np = rendering_unshadowed.cpu().detach().numpy().transpose(1, 2, 0)
        shadowed_np = rendering_shadowed.cpu().detach().numpy().transpose(1, 2, 0)
        gt_image_np = gt_image.cpu().detach().numpy().transpose(1, 2, 0)
        
        _, full = ssim_skimage(unshadowed_np, gt_image_np, win_size=5, channel_axis=2, full=True, data_range=1.0)
        mssim_over_mask = (torch.tensor(full).cuda()*mask.unsqueeze(-1)).sum() / (3*mask.sum())
        ssims_unshadowed_scikit.append(mssim_over_mask)
        _, full = ssim_skimage(shadowed_np, gt_image_np, win_size=5, channel_axis=2, full=True, data_range=1.0)
        mssim_over_mask = (torch.tensor(full).cuda()*mask.unsqueeze(-1)).sum() / (3*mask.sum())
        ssims_shadowed_scikit.append(mssim_over_mask)

    # save metrics 
    with open(os.path.join(render_path, "metrics.txt"), 'w') as f:
        print("  PSNR unshadowed: {:>12.7f}".format(torch.tensor(psnrs_unshadowed).mean(), ".5"), file=f)
        print("  MSE unshadowed: {:>12.7f}".format(torch.tensor(mse_unshadowed).mean(), ".5"), file=f)
        print("  MAE unshadowed: {:>12.7f}".format(torch.tensor(mae_unshadowed).mean(), ".5"), file=f)
        print("  SSIM skimage unshadowed: {:>12.7f}".format(torch.tensor(ssims_unshadowed_scikit).mean(), ".5"), file=f)

        print("  PSNR shadowed: {:>12.7f}".format(torch.tensor(psnrs_shadowed).mean(), ".5"), file=f)
        print("  MSE shadowed: {:>12.7f}".format(torch.tensor(mse_shadowed).mean(), ".5"), file=f)
        print("  MAE shadowed: {:>12.7f}".format(torch.tensor(mae_shadowed).mean(), ".5"), file=f)
        print("  SSIM skimage shadowed: {:>12.7f}".format(torch.tensor(ssims_shadowed_scikit).mean(), ".5"), file=f)

        print(f" best psnrs, image order: {psnrs_shadowed}. optimized for psnr shadowed", file=f)

    print("PSNR MEAN {:>12.7f}".format(torch.tensor(psnrs_shadowed).mean(), ".5"))
    print("MSE shadowed: {:>12.7f}".format(torch.tensor(mse_shadowed).mean(), ".5"))
    print("MAE shadowed: {:>12.7f}".format(torch.tensor(mae_shadowed).mean(), ".5"))
    print("SSIM skimage shadowed: {:>12.7f}".format(torch.tensor(ssims_shadowed_scikit).mean(), ".5"))

    
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--test_config", default="", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_set(model.extract(args), args.iteration, pipeline.extract(args), opt.extract(args), args.test_config)