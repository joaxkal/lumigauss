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

# import torch

# def mse(img1, img2):
#     return (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)

# def psnr(img1, img2):
#     mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))

import torch

def mse(img1, img2, mask=None):
    if torch.is_tensor(mask):
        diff = (img1 - img2) ** 2
        diff = diff * mask
        mse_value = diff.reshape(img1.shape[0], -1).sum(1, keepdim=True) / (mask.reshape(img1.shape[0], -1).sum(1, keepdim=True)+1e-6)
    else:
        mse_value = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse_value

def psnr(img1, img2, mask=None):
    mse_value = mse(img1, img2, mask)
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse_value))
    return psnr_value
