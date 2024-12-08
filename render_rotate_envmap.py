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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import json
import imageio
from utils.sh_vis_utils import shReconstructDiffuseMap
from utils.sh_rotate_utils import Rotation
from utils.normal_utils import compute_normal_world_space
import cv2


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
thickness = 2
line_type = cv2.LINE_AA

def rgb2greyscale(image):
    """
    Convert RGB image to grayscale
    """
    ret_image = 0.2126 * image[:, 0, :, :] + 0.7152 * image[:, 1, :, :] + 0.0722 * image[:, 2, :, :]
    ret_image = ret_image[:, None, :, :]
    ret_image = ret_image.repeat(1, 3, 1, 1)
    return ret_image

def create_video_for_renders(output_path, shadowed_renders, unshadowed_renders, shadows, illumination):
    
    writer = imageio.get_writer(output_path, fps=15)

    for i in range(min(len(illumination), len(shadows), len(unshadowed_renders), len(shadowed_renders))):
        
        img_illumination = (np.array(illumination[i]) * 255).clip(0, 255).astype(np.uint8)
        img_shadows = (np.array(shadows[i]) * 255).clip(0, 255).astype(np.uint8)
        img_unshadowed = (np.array(unshadowed_renders[i]) * 255).clip(0, 255).astype(np.uint8)
        img_shadowed = (np.array(shadowed_renders[i]) * 255).clip(0, 255).astype(np.uint8)

        h, w, c = img_illumination.shape

        composite_frame = np.zeros((2 * h, 2 * w, c), dtype=np.uint8)
        composite_frame[0:h, 0:w] = img_illumination
        composite_frame[0:h, w:2*w] = img_shadows
        composite_frame[h:2*h, 0:w] = img_unshadowed
        composite_frame[h:2*h, w:2*w] = img_shadowed

        cv2.putText(composite_frame, 'Illumination', (10, 30), font, font_scale, font_color, thickness, line_type)
        cv2.putText(composite_frame, 'Shadows', (w + 10, 30), font, font_scale, font_color, thickness, line_type)
        cv2.putText(composite_frame, 'Unshadowed', (10, h + 30), font, font_scale, font_color, thickness, line_type)
        cv2.putText(composite_frame, 'Shadowed', (w + 10, h + 30), font, font_scale, font_color, thickness, line_type)

        writer.append_data(composite_frame)

    writer.close()

def create_video_for_envmap(output_path, envs):
    writer_map = imageio.get_writer(output_path, fps=15)
    for env in envs:
        new_env_img = shReconstructDiffuseMap(env, width=600)
        new_env_img = torch.tensor(new_env_img**(1/ 2.2))
        new_env_img = np.array(new_env_img*255).clip(0,255).astype(np.uint8)
        writer_map.append_data(np.array(new_env_img))
    writer_map.close()


def render_set(dataset : ModelParams, iteration : int, pipeline : PipelineParams, envmaps, viewpoints):

    with torch.no_grad():
        dataset.eval = False
        gaussians = GaussianModel(dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a)
        scene = Scene(dataset, gaussians, load_iteration=iteration)
            
        if gaussians.with_mlp:
            gaussians.mlp.eval()
            gaussians.embedding.eval()
        
        with open(os.path.join(dataset.model_path, "appearance_lut.json")) as handle:
            appearance_lut = json.loads(handle.read())

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
       
        cameras_tmp = scene.getTrainCameras()
        cameras = [c for c in cameras_tmp if any(n in c.image_name for n in viewpoints)]
        assert len(cameras)>0
        
        # get train + external envmaps
        env_map_list = []

        for c in cameras:
            appearance_idx = appearance_lut.get(c.image_name)
            if appearance_idx is not None:
                env_map_list.append(gaussians.compute_env_sh(appearance_idx).cpu().numpy().T)
            else:
                print(f"Warning: {c.image_name} not found in training envmap list.")

        for env_file in sorted(os.listdir(envmaps)):
            env_temp = np.loadtxt(os.path.join(envmaps, env_file))
            env_map_list.append(env_temp)

        # angle interpolation
        interp_steps = 30
        line_points = np.linspace(0, 1, interp_steps)
        angle_start = 0
        angle_end = 3.14*2
        interp_angle = np.interp(line_points, [0, 1], [angle_start, angle_end])

        for axis in [0]: # rotate only around y

            render_path_parent = os.path.join(dataset.model_path, "rotate_env_map", "ours_{}".format(scene.loaded_iter))
            render_path = os.path.join(render_path_parent, f"renders_axis{axis}")
            makedirs(render_path, exist_ok=True)
                        
            for env_sh_idx, env_sh in enumerate(env_map_list):

                rot_envs_list = []
                renders_unshadowed = {i: [] for i in range(len(cameras))}
                renders_shadowed = {i: [] for i in range(len(cameras))}
                renders_lumi_diff = {i: [] for i in range(len(cameras))}
                renders_illumination = {i: [] for i in range(len(cameras))}
                
                for angle in tqdm(interp_angle, desc=f"Rotating envmap {env_sh_idx}"):
                    angle_vec = np.array([0, 0, 0], dtype=np.float32)
                    angle_vec[axis] = angle.item()

                    rotation = Rotation()
                    rot = np.float32(np.dot(rotation.rot_y(angle_vec[0]), np.dot(rotation.rot_x(angle_vec[1]), rotation.rot_z(angle_vec[2]))))
                    rot_env_sh = np.matmul(rot, env_sh)
                    rot_envs_list.append(rot_env_sh)
                    rot_env_sh_torch=torch.tensor(rot_env_sh.T, dtype=torch.float32).cuda().cuda()

                    for render_idx, view in enumerate(cameras):
                        
                        #get normals in world space
                        quaternions = gaussians.get_rotation
                        scales = gaussians.get_scaling
                        normal_vectors, multiplier = compute_normal_world_space(quaternions, scales, view.world_view_transform, gaussians.get_xyz)
       
                        #albedo
                        rgb_precomp_alb =gaussians.get_albedo
                        render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_alb)
                        albedo = render_pkg["render"]
                        
                        # unshadowed relightning
                        rgb_precomp_unshadowed, _ =gaussians.compute_gaussian_rgb(rot_env_sh_torch, shadowed=False, normal_vectors=normal_vectors, env_hemisphere_lightning=True)
                        render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_unshadowed)
                        rendering = render_pkg["render"]
                        renders_unshadowed[render_idx].append(rendering.permute(1,2,0).cpu().numpy())
                        
                        # unshadowed irradiance
                        lum_unshadowed=rendering /albedo
                        
                        # shadowed relightning
                        rgb_precomp, _ = gaussians.compute_gaussian_rgb(rot_env_sh_torch, multiplier=multiplier)
                        rendering = render(view, gaussians, pipeline, background, override_color=rgb_precomp)["render"]
                        renders_shadowed[render_idx].append(rendering.permute(1,2,0).cpu().numpy())
                        
                        # shadowed irradiance
                        lum_shadowed=rendering /albedo
                        
                        # Compute luminance difference
                        luminance_diff = rgb2greyscale((lum_unshadowed - lum_shadowed).unsqueeze(0)).squeeze()
                        luminance_diff = 1 - torch.clamp(luminance_diff, min=0, max=1000)
                        renders_lumi_diff[render_idx].append(luminance_diff.permute(1,2,0).cpu().numpy())

                        # add illuminance to renders
                        if lum_unshadowed.max()>1:
                            lum = lum_unshadowed/lum_unshadowed.max()
                        else:
                            lum = lum_unshadowed
                        renders_illumination[render_idx].append(lum.permute(1,2,0).cpu().numpy())


                # Create videos
                output_path = os.path.join(render_path, f'rotate_map{env_sh_idx}.mp4')
                create_video_for_envmap(output_path, rot_envs_list)
                
                for idx in range(len(cameras)):
                    output_path = os.path.join(render_path, f'map{env_sh_idx}_campos{idx}.mp4')
                    create_video_for_renders(output_path, 
                                             unshadowed_renders=renders_unshadowed[idx], shadowed_renders=renders_shadowed[idx],
                                             shadows=renders_lumi_diff[idx], illumination=renders_illumination[idx])
                    
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--envmaps", type=str, required=True, 
                        help="Path to folder with environment map files (in .txt)")
    parser.add_argument("--viewpoints", type=str, nargs='+', required=True, 
                        help="List of image names (e.g., IMAGE1.JPG IMAGE2.JPG)")


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_set(model.extract(args), args.iteration, pipeline.extract(args), args.envmaps, args.viewpoints)
