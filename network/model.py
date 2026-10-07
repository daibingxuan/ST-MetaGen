import torch
import torch.nn.functional as F
from tqdm import tqdm
from network.model_utils import *
from network.unet import UNetModel
from einops import rearrange, repeat
import numpy as np
from random import random
from functools import partial
from torch import nn
from torch.special import expm1
import sys
import joblib
import pdb
import math
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt

TRUNCATED_TIME = 0.1

def get_symmetry_operations(space_group: int):
    """根据空间群返回对应的对称操作列表（3D张量变换函数）"""
    # 立方晶系基础对称操作示例，可按空间群扩展
    ops = []
    # 90° 绕x轴旋转
    ops.append(lambda x: torch.rot90(x, k=1, dims=(2,3)))
    # 90° 绕y轴旋转
    ops.append(lambda x: torch.rot90(x, k=1, dims=(3,4)))
    # 90° 绕z轴旋转
    ops.append(lambda x: torch.rot90(x, k=1, dims=(2,4)))
    # 中心反演
    ops.append(lambda x: torch.flip(x, dims=(2,3,4)))
    return ops


class OccupancyDiffusion(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            base_channels: int = 128,
            attention_resolutions: str = "16,8",
            with_attention: bool = False,
            num_heads: int = 4,
            dropout: float = 0.0,
            verbose: bool = False,
            use_geo_condition: bool = True,
            geo_condition_dim: int = 50,
            use_tensor_condition: bool = False,
            tensor_condition_dim: int = 3,
            eps: float = 1e-6,
            noise_schedule: str = "linear",
    ):
        super().__init__()
        self.image_size = image_size
        if image_size == 8:
            channel_mult = (1, 4, 8)
        elif image_size == 32:
            channel_mult = (1, 2, 4, 8)
        elif image_size == 64:
            channel_mult = (1, 2, 4, 8, 8)
        else:
            raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        self.eps = eps
        self.verbose = verbose
        self.use_geo_condition = use_geo_condition
        self.use_tensor_condition = use_tensor_condition
        # 保存条件维度，用于构造对应维度的全零缺失张量
        self.geo_condition_dim = geo_condition_dim
        self.tensor_condition_dim = tensor_condition_dim

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.denoise_fn = UNetModel(
            image_size=image_size,
            base_channels=base_channels,
            dim_mults=channel_mult,
            dropout=dropout,
            use_geo_condition=use_geo_condition,
            geo_condition_dim=geo_condition_dim,
            use_tensor_condition=use_tensor_condition,
            tensor_condition_dim=tensor_condition_dim,
            world_dims=3,
            num_heads=num_heads,
            attention_resolutions=tuple(attention_ds),
            with_attention=with_attention,
            verbose=verbose
        )

        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def get_sampling_timesteps_uneven(self, batch, device, steps):
        steps1 = 10
        steps2 = 8
        times1 = torch.linspace(1.0, 0.8, steps1 + 1, device=device)
        times2 = torch.linspace(0.8, 0.0, steps2 + 1, device=device)
        times = torch.cat((times1[:-1], times2))
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def compute_lambda(self, step, max_steps, start_value=0.0, end_value=1.0):
        return start_value + (end_value - start_value) * (step / max_steps)

    def training_loss(self, img, geo_condition, prop, step, max_steps, *args, **kwargs):
        batch = img.shape[0]

        times = torch.zeros((batch,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise

        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.denoise_fn(
                    noised_img, noise_level, geo_condition, prop
                ).detach_()

        pred = self.denoise_fn(noised_img, noise_level, geo_condition, prop, self_cond)

        # 主损失：MSE 预测干净体素
        noise_loss = F.mse_loss(pred, img)

        # # 辅助损失：二分类 BCE
        # pred_bin = (pred > 0).float()
        # img_bin = (img > 0).float()
        # bce_loss = self.bce_loss_fn(pred, img_bin)

        # lambda_bce = self.compute_lambda(step, max_steps, start_value=0.0, end_value=1.0)
        # total_loss = noise_loss + 0.3 * bce_loss

        # 返回总损失用于反向传播
        return noise_loss

    @torch.no_grad()
    def sample_unconditional(self, batch_size=16, steps=50, truncated_index=0.0, verbose=True):
        """
        纯无条件生成：几何+属性双条件均为全零张量
        与训练时「双条件整体缺失」的分布完全对齐
        """
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        batch, device = shape[0], self.device
        time_pairs = self.get_sampling_timesteps_uneven(batch, device=device, steps=steps)

        img = torch.randn(shape, device=device)
        x_start = None

        # 全零张量表示缺失，始终走编码器路径，与训练场景严格一致
        geo_zero = torch.zeros(batch, self.geo_condition_dim, device=device) if self.use_geo_condition else None
        prop_zero = torch.zeros(batch, self.tensor_condition_dim, device=device) if self.use_tensor_condition else None

        _iter = tqdm(time_pairs, desc='sampling loop time step') if verbose else time_pairs
        for time, time_next in _iter:
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            x_start = self.denoise_fn(img, log_snr, geo_zero, prop_zero, x_start)
            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
            img = x_start * alpha_next + pred_noise * sigma_next

        return img

    @torch.no_grad()
    def sample_with_tensor(self, tensor_c, batch_size=16, steps=50, truncated_index=0.0, tensor_w=1.0, verbose=True):
        """
        仅属性条件引导生成：几何条件整体缺失（全零），属性条件做CFG引导
        支持属性单维度缺失：对应维度填0即可，与训练分布完全对齐
        """
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        batch, device = shape[0], self.device
        time_pairs = self.get_sampling_timesteps(batch, device=device, steps=steps)

        # 属性条件处理
        tensor_condition = torch.from_numpy(np.asarray(tensor_c).squeeze().astype(np.float32)).to(device)
        if tensor_condition.ndim == 1:
            tensor_condition = tensor_condition.unsqueeze(0).repeat(batch, 1)
        tensor_zero = torch.zeros_like(tensor_condition)

        # 几何条件全程传全零张量（整体缺失），不走None分支，与训练对齐
        geo_zero = torch.zeros(batch, self.geo_condition_dim, device=device) if self.use_geo_condition else None

        img = torch.randn(shape, device=device)
        x_start = None

        _iter = tqdm(time_pairs, desc='sampling loop time step') if verbose else time_pairs
        for time, time_next in _iter:
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            x_zero_none = self.denoise_fn(img, log_snr, geo_zero, tensor_zero, x_start)
            x_cond = self.denoise_fn(img, log_snr, geo_zero, tensor_condition, x_start)
            x_start = x_zero_none + tensor_w * (x_cond - x_zero_none)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
            img = x_start * alpha_next + pred_noise * sigma_next
        return img


    @torch.no_grad()
    def sample_with_all_condition(self, geo_condition, prop=None, batch_size=16,
                                steps=50, truncated_index=0.0, 
                                cond_w=None, w_geo=1.5, w_prop=1.5,
                                # ========== 仅体积分数约束参数 ==========
                                use_vf_constraint=False, vf_target=None, w_vf=0.3,
                                verbose=True):
        # 兼容旧版单权重
        if cond_w is not None:
            w_geo = cond_w
            w_prop = cond_w

        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        batch, device = shape[0], self.device
        time_pairs = self.get_sampling_timesteps(batch, device=device, steps=steps)

        # 几何条件预处理
        geo_cond_np = np.asarray(geo_condition.detach().cpu().numpy()).squeeze().astype(np.float32)
        if geo_cond_np.ndim == 1:
            geo_cond = torch.from_numpy(geo_cond_np).to(device).unsqueeze(0).repeat(batch, 1)
        else:
            geo_cond = torch.from_numpy(geo_cond_np).to(device)
        geo_zero = torch.zeros_like(geo_cond)

        # 属性条件预处理
        if self.use_tensor_condition:
            if prop is not None:
                prop_np = np.asarray(prop.detach().cpu().numpy()).squeeze().astype(np.float32)
                if prop_np.ndim == 1:
                    prop_cond = torch.from_numpy(prop_np).to(device).unsqueeze(0).repeat(batch, 1)
                else:
                    prop_cond = torch.from_numpy(prop_np).to(device)
            else:
                prop_cond = torch.zeros(batch, self.tensor_condition_dim, device=device)
            prop_zero = torch.zeros_like(prop_cond)
        else:
            prop_cond = None
            prop_zero = None

        img = torch.randn(shape, device=device)
        x_start = None

        _iter = tqdm(time_pairs, desc='sampling loop time step') if verbose else time_pairs
        for time, time_next in _iter:
            log_snr_1d = self.log_snr(time)
            log_snr_next_1d = self.log_snr(time_next)
            
            log_snr = right_pad_dims_to(img, log_snr_1d)
            log_snr_next = right_pad_dims_to(img, log_snr_next_1d)

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # ========== 双权重CFG ==========
            x_uncond = self.denoise_fn(img, log_snr_1d, geo_zero, prop_zero, x_start)
            x_geo_only = self.denoise_fn(img, log_snr_1d, geo_cond, prop_zero, x_start)
            x_prop_only = self.denoise_fn(img, log_snr_1d, geo_zero, prop_cond, x_start)
            
            x_start = x_uncond \
                    + w_geo * (x_geo_only - x_uncond) \
                    + w_prop * (x_prop_only - x_uncond)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
            img = alpha_next * x_start + sigma_next * pred_noise

        return img
    @torch.no_grad()
    def sample_with_geo_condition(self, geo_condition, prop=None, batch_size=16,
                                steps=50, truncated_index=0.0, cond_w=1.0, verbose=True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        batch, device = shape[0], self.device
        time_pairs = self.get_sampling_timesteps(batch, device=device, steps=steps)

        # ========== 修正1：geo_condition 统一处理为 (batch, 50) 二维 ==========
        geo_cond_np = np.asarray(geo_condition).squeeze().astype(np.float32)
        if geo_cond_np.ndim == 1:
            # 单条一维输入 → 复制batch份，和sample_with_tensor逻辑一致
            geo_cond = torch.from_numpy(geo_cond_np).to(device).unsqueeze(0).repeat(batch, 1)
        else:
            # 已经是二维 → 直接使用（确保第一维等于batch_size）
            geo_cond = torch.from_numpy(geo_cond_np).to(device)
        geo_zero = torch.zeros_like(geo_cond)  # 无条件用全零向量

        # prop 条件处理（保持原逻辑）
        if prop is not None:
            prop_np = np.asarray(prop).squeeze().astype(np.float32)
            if prop_np.ndim == 1:
                prop_cond = torch.from_numpy(prop_np).to(device).unsqueeze(0).repeat(batch, 1)
            else:
                prop_cond = torch.from_numpy(prop_np).to(device)
            prop_zero = torch.zeros_like(prop_cond)
        else:
            prop_cond = None
            prop_zero = None

        img = torch.randn(shape, device=device)
        x_start = None

        _iter = tqdm(time_pairs, desc='sampling loop time step') if verbose else time_pairs
        for time, time_next in _iter:
            log_snr_1d = self.log_snr(time)
            log_snr_next_1d = self.log_snr(time_next)
            
            log_snr = right_pad_dims_to(img, log_snr_1d)
            log_snr_next = right_pad_dims_to(img, log_snr_next_1d)

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            x_zero_none = self.denoise_fn(img, log_snr_1d, geo_zero, prop_zero, x_start)
            x_cond = self.denoise_fn(img, log_snr_1d, geo_cond, prop_cond, x_start)

            x_start = x_zero_none + cond_w * (x_cond - x_zero_none)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
            img = x_start * alpha_next + pred_noise * sigma_next
        return img
