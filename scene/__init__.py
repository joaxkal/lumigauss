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
import glob
import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        split_file_pattern = os.path.join(args.source_path, "*split.csv")
        split_files = glob.glob(split_file_pattern)

        if split_files:
            args.eval_file = split_files[0]
            print(f"SPLIT FILE USED: {args.eval_file}")
        else:
            args.eval_file = None
            print("WARNING: No split file found. Please check if one exists in the directory or modify scene/init.py.")

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.eval_file)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.eval_file)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](
                args.source_path, "cameras_sphere.npz", "cameras_sphere.npz", eval=args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if shuffle:
            random.Random(567464).shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.Random(774452).shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras", len(scene_info.train_cameras))
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras", len(scene_info.test_cameras))
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            if self.gaussians.with_mlp:
                self.gaussians.mlp.load_state_dict(torch.load(self.model_path + "/chkpnt_mlp" + str(self.loaded_iter) + ".pth", weights_only=True))
                self.gaussians.embedding.load_state_dict(torch.load(self.model_path + "/chkpnt_embedding" + str(self.loaded_iter) + ".pth", weights_only=True))
            else:
                self.gaussians.env_params.load_state_dict(torch.load(self.model_path + "/chkpnt_env" + str(self.loaded_iter) + ".pth", weights_only=True))

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians.with_mlp:
            torch.save(self.gaussians.mlp.state_dict(), self.model_path + "/chkpnt_mlp" + str(iteration) + ".pth")
            torch.save(self.gaussians.embedding.state_dict(), self.model_path + "/chkpnt_embedding" + str(iteration) + ".pth")
        else:
            torch.save(self.gaussians.env_params.state_dict(), self.model_path + "/chkpnt_env" + str(iteration) + ".pth")


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]