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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.sh_utils import eval_sh_point

TINY_NUMBER = 1e-6

def l1_loss(network_output, gt, mask=None):
    if torch.is_tensor(mask):
        return ((torch.abs((network_output*mask - gt*mask))).sum()) / (network_output.shape[0]*mask.sum() + 1e-6 )
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt, mask):
    if torch.is_tensor(mask):
        return (((network_output*mask - gt*mask) ** 2).sum()) / (network_output.shape[0]*mask.sum() + 1e-6 )
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        if torch.is_tensor(mask):
            return (ssim_map*mask).sum() / (ssim_map.shape[0]*mask.sum() + 1e-6 )
        return ssim_map.mean()
    else:
        if torch.is_tensor(mask):
            raise NotImplemented
        return ssim_map.mean(1).mean(1).mean(1)


def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(0)) / (torch.sum(mask) * x.shape[0] + TINY_NUMBER)

def img2mae(x, y, mask=None):
    if mask is None:
        return torch.mean(torch.abs(x - y))
    else:
        return torch.sum(torch.abs(x - y) * mask.unsqueeze(0)) / (torch.sum(mask) * x.shape[0] + TINY_NUMBER)

def mse2psnr(x):
    return -10. * torch.log(torch.tensor(x)+TINY_NUMBER) / torch.log(torch.tensor(10))

def img2mse_image(x, y, mask=None):
    
    # Compute squared difference per pixel per channel
    mse_image = (x - y) ** 2
    
    # If a mask is provided, apply the mask
    if mask is not None:
        # Ensure the mask is expanded to match the shape (3, W, H)
        mask = mask.unsqueeze(0)  # Add channel dimension (1, W, H) -> (3, W, H)
        mse_image = mse_image * mask

    return mse_image


def penalize_outside_range(tensor, lower_bound=0.0, upper_bound=1.0):
    error = 0
    below_lower_bound = tensor[tensor < lower_bound]
    above_upper_bound = tensor[tensor > upper_bound]
    if below_lower_bound.numel():
        error += torch.mean((below_lower_bound - lower_bound) ** 2)
    if above_upper_bound.numel():
        error += torch.mean((above_upper_bound - upper_bound) ** 2)
    return error

def compute_sh_gauss_losses(shs_gauss, normals, N_samples=25):

    normal_vectors_expand = normals.repeat_interleave(N_samples, dim=0)
    shs_gauss_expand = shs_gauss.repeat_interleave(N_samples, dim=0)
    
    view_dir_unnorm =torch.empty(shs_gauss_expand.shape[0], 3, device=shs_gauss_expand.device).uniform_(-1,1)
    view_dir = view_dir_unnorm / (view_dir_unnorm.norm(dim=1, keepdim=True)+0.0000001)
    
    # eval SH_gauss for each sampled dir
    sampled_shs_gauss = eval_sh_point(view_dir, shs_gauss_expand[:,0:1,:]) # we only have 0th sh channel for gaussian - shadow, no color bleeding

    #compute difference for shadowed and unshadowed values
    dot_product_n_v_raw = torch.sum(view_dir * normal_vectors_expand, dim=1, keepdim=True)
    dot_product_n_v = torch.clamp(dot_product_n_v_raw, min=0)
    shadowed_unshadowed_diff = sampled_shs_gauss - dot_product_n_v

    # SHgauss has to be in range 0-1, eq. 12
    sh_gauss_loss = penalize_outside_range(sampled_shs_gauss.view(-1), 0.0, 1.0)

    # Shadowed SH and unshadowed values need to be consistent, eq.14: ||eval_sh(view_dir) - max(0, dot(view_dir, normal))||^2
    consistency_loss = torch.mean(shadowed_unshadowed_diff ** 2)

    # Enforce shadowed SH to have lower values than unshadowed, eq. 15
    shadow_loss = (torch.clamp(shadowed_unshadowed_diff, min=0.0)**2).mean()
    
    return sh_gauss_loss, consistency_loss, shadow_loss

def compute_sh_env_loss(sh_env, N_samples=10):

    shs_view = sh_env.unsqueeze(0).permute(0, 2, 1).repeat(N_samples, 1, 1)
    view_dir_unnorm =torch.empty(shs_view.shape[0], 3, device=shs_view.device).uniform_(-1,1)
    view_dir = view_dir_unnorm / view_dir_unnorm.norm(dim=1, keepdim=True)
    sampled_shs_gauss = eval_sh_point(view_dir, shs_view)

    #SH env has to be in R+, eq. 13 
    sh_env_loss = penalize_outside_range(sampled_shs_gauss.view(-1), 0.0,torch.inf)
    
    return sh_env_loss