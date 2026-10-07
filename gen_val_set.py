import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from bitstring import BitArray
import argparse
import random
from scipy.ndimage import label
from network.model import OccupancyDiffusion

# ===================== 全局常量 与 工具函数 =====================
TRUNCATED_TIME = 0.1

def split_random_indices(total_samples: int, val_ratio=0.2, seed=42):
    random.seed(seed)
    indices = list(range(total_samples))
    random.shuffle(indices)
    split_point = int(total_samples * (1 - val_ratio))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    return train_indices, val_indices

def save_voxel(voxel: np.ndarray, save_path: str):
    flat_voxel = np.ravel(voxel).astype(bool)
    bits = BitArray(bin="".join(np.where(flat_voxel, "1", "0")))
    reordered_bits = BitArray()
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        reordered_bits.append(byte[::-1])
    with open(save_path, 'wb') as f:
        reordered_bits.tofile(f)

def voxel_postprocess(
    voxel: np.ndarray,
    enforce_periodicity: bool = True,
    keep_largest_component: bool = True,
    remove_small_solids: bool = True,
    remove_small_holes: bool = True,
    min_solid_size: int = 10,
    min_hole_size: int = 10
) -> np.ndarray:
    struct_26 = np.ones((3, 3, 3), dtype=bool)
    out = voxel.copy()
    if enforce_periodicity:
        boundary_x = np.logical_and(out[0], out[-1])
        out[0] = boundary_x
        out[-1] = boundary_x
        boundary_y = np.logical_and(out[:, 0, :], out[:, -1, :])
        out[:, 0, :] = boundary_y
        out[:, -1, :] = boundary_y
        boundary_z = np.logical_and(out[:, :, 0], out[:, :, -1])
        out[:, 0] = boundary_z
        out[:, -1] = boundary_z
    if remove_small_solids:
        labeled, num = label(out, structure=struct_26)
        if num > 1:
            sizes = np.bincount(labeled.flat)[1:]
            keep_mask = sizes >= min_solid_size
            keep_labels = np.where(keep_mask)[0] + 1
            out = np.isin(labeled, keep_labels)
    if keep_largest_component:
        labeled, num = label(out, structure=struct_26)
        if num > 1:
            sizes = np.bincount(labeled.flat)[1:]
            max_label = np.argmax(sizes) + 1
            out = (labeled == max_label)
    if remove_small_holes:
        background = ~out
        labeled_bg, num_bg = label(background, structure=struct_26)
        if num_bg > 1:
            sizes_bg = np.bincount(labeled_bg.flat)[1:]
            edge_labels = set()
            edge_labels.update(labeled_bg[0, :, :].flatten())
            edge_labels.update(labeled_bg[-1, :, :].flatten())
            edge_labels.update(labeled_bg[:, 0, :].flatten())
            edge_labels.update(labeled_bg[:, -1, :].flatten())
            edge_labels.update(labeled_bg[:, :, 0].flatten())
            edge_labels.update(labeled_bg[:, :, -1].flatten())
            for lbl in range(1, num_bg + 1):
                if lbl not in edge_labels and sizes_bg[lbl-1] < min_hole_size:
                    out[labeled_bg == lbl] = True
    return out

# ===================== 验证集Dataset =====================
class VoxelValDataset(Dataset):
    def __init__(
        self,
        full_voxels: np.ndarray,
        full_geo_cond: np.ndarray,
        full_labels: np.ndarray,
        full_prop: np.ndarray,
        idx_list: list
    ):
        self.indices = idx_list
        self.voxels = full_voxels[idx_list].astype(np.float32)
        self.geo_conditions = full_geo_cond[idx_list].astype(np.float32)
        self.labels = full_labels[idx_list].astype(np.float32)
        self.prop = full_prop[idx_list] if full_prop is not None else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        res = {}
        voxel = torch.tensor(self.voxels[idx])
        voxel = 2 * voxel - 1
        voxel = voxel.unsqueeze(0)
        res["occupancy"] = voxel

        geo_cond = torch.tensor(self.geo_conditions[idx], dtype=torch.float32)
        res["condition"] = geo_cond

        res["label"] = torch.tensor(self.labels[idx])
        res["idx"] = self.indices[idx]

        if self.prop is not None:
            prop = torch.tensor(self.prop[idx], dtype=torch.float32)
            res["tensor_feature"] = prop
        else:
            res["tensor_feature"] = torch.zeros(3)
        return res

@torch.no_grad()
def sample_smc_cfg(
    generator,
    geo_condition,
    prop=None,
    batch_size=16,
    steps=50,
    truncated_time=0.1,
    cond_w=None,
    w_geo=1.5,
    w_prop=1.5,
    smc_cfg_enable: bool = True,
    smc_cfg_lambda: float = 5.0,
    smc_cfg_K: float = 0.2,
    no_cfg_warmup_steps: int = 2,
    use_saturation: bool = False,
    saturation_threshold: float = 0.1,
    use_smc_for_prop: bool = False,
    adaptive_K: bool = False,
    K_min: float = 0.1,
    K_max: float = 0.5,
    verbose: bool = True,
):
    """
    SMC-CFG采样函数 - 纯函数实现，与模型权重无关
    """
    if cond_w is not None:
        w_geo = cond_w
        w_prop = cond_w

    image_size = generator.image_size
    shape = (batch_size, 1, image_size, image_size, image_size)
    batch, device = shape[0], generator.device
    
    # 获取时间步
    time_pairs = generator.get_sampling_timesteps(batch, device=device, steps=steps)

    # 条件准备
    if hasattr(generator, '_to_condition_tensor'):
        geo_cond = generator._to_condition_tensor(geo_condition, batch, device, generator.geo_condition_dim)
    else:
        geo_cond = torch.as_tensor(geo_condition, device=device, dtype=torch.float32)
        if geo_cond.ndim == 1:
            geo_cond = geo_cond.unsqueeze(0).repeat(batch, 1)
    geo_zero = torch.zeros_like(geo_cond)

    use_tensor_condition = getattr(generator, 'use_tensor_condition', False)
    if use_tensor_condition:
        if prop is not None:
            if hasattr(generator, '_to_condition_tensor'):
                prop_cond = generator._to_condition_tensor(prop, batch, device, generator.tensor_condition_dim)
            else:
                prop_cond = torch.as_tensor(prop, device=device, dtype=torch.float32)
                if prop_cond.ndim == 1:
                    prop_cond = prop_cond.unsqueeze(0).repeat(batch, 1)
        else:
            prop_cond = torch.zeros(batch, generator.tensor_condition_dim, device=device)
        prop_zero = torch.zeros_like(prop_cond)
    else:
        prop_cond = None
        prop_zero = None

    # SMC状态变量
    e_prev_geo = None
    e_prev_prop = None
    current_K = smc_cfg_K  # 保持标量

    # 初始化
    x_t = torch.randn(shape, device=device)
    x_self_cond = None

    from network.model_utils import right_pad_dims_to, log_snr_to_alpha_sigma

    _iter = tqdm(time_pairs, desc='SMC-CFG Sampling') if verbose else time_pairs

    for step_idx, (time, time_next) in enumerate(_iter):
        log_snr_1d = generator.log_snr(time)
        log_snr_next_1d = generator.log_snr(time_next)
        log_snr = right_pad_dims_to(x_t, log_snr_1d)
        log_snr_next = right_pad_dims_to(x_t, log_snr_next_1d)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # ===== Step 1: CFG预测 =====
        x_uncond = generator.denoise_fn(x_t, log_snr_1d, geo_zero, prop_zero, x_self_cond)
        x_geo_only = generator.denoise_fn(x_t, log_snr_1d, geo_cond, prop_zero, x_self_cond)
        e_t_geo = x_geo_only - x_uncond
        # ===== Step 2: SMC-CFG控制 =====
        if smc_cfg_enable and step_idx >= no_cfg_warmup_steps:
            if e_prev_geo is not None:
                # 滑模面
                s_t_geo = (e_t_geo - e_prev_geo) + smc_cfg_lambda * e_prev_geo
                # 自适应增益
                if adaptive_K:
                    # 计算每个样本的误差范数，然后扩展到与s_t_geo相同的维度
                    error_norm = torch.norm(e_t_geo.flatten(start_dim=1), dim=1, keepdim=True)
                    error_norm = error_norm / (error_norm.mean() + 1e-6)
                    # current_K变成[batch, 1, 1, 1, 1]以便广播
                    current_K_per_sample = torch.clamp(smc_cfg_K * error_norm, K_min, K_max)
                    # 扩展维度以匹配s_t_geo: [batch, 1, 1, 1, 1]
                    current_K_expanded = current_K_per_sample.view(-1, 1, 1, 1, 1)
                else:
                    # 标量K，直接使用
                    current_K_expanded = smc_cfg_K
                # 切换控制
                if use_saturation:
                    if adaptive_K:
                        # current_K_expanded是[batch, 1, 1, 1, 1]，可以广播
                        u_sw_geo = -current_K_expanded * torch.clamp(s_t_geo / saturation_threshold, -1.0, 1.0)
                    else:
                        u_sw_geo = -current_K_expanded * torch.clamp(s_t_geo / saturation_threshold, -1.0, 1.0)
                else:
                    if adaptive_K:
                        u_sw_geo = -current_K_expanded * torch.sign(s_t_geo)
                    else:
                        u_sw_geo = -current_K_expanded * torch.sign(s_t_geo)
                
                x_geo_controlled = x_uncond + w_geo * e_t_geo + u_sw_geo
            else:
                x_geo_controlled = x_uncond + w_geo * e_t_geo
        else:
            x_geo_controlled = x_uncond + w_geo * e_t_geo
        e_prev_geo = e_t_geo.detach().clone()

        # ===== Step 3: 属性条件 =====
        if use_tensor_condition:
            x_prop_only = generator.denoise_fn(x_t, log_snr_1d, geo_zero, prop_cond, x_self_cond)
            e_t_prop = x_prop_only - x_uncond
            if smc_cfg_enable and step_idx >= no_cfg_warmup_steps and use_smc_for_prop:
                if e_prev_prop is not None:
                    s_t_prop = (e_t_prop - e_prev_prop) + smc_cfg_lambda * e_prev_prop
                    if adaptive_K:
                        # 使用同样的K
                        if use_saturation:
                            u_sw_prop = -current_K_expanded * torch.clamp(s_t_prop / saturation_threshold, -1.0, 1.0)
                        else:
                            u_sw_prop = -current_K_expanded * torch.sign(s_t_prop)
                    else:
                        if use_saturation:
                            u_sw_prop = -smc_cfg_K * torch.clamp(s_t_prop / saturation_threshold, -1.0, 1.0)
                        else:
                            u_sw_prop = -smc_cfg_K * torch.sign(s_t_prop)
                    x_prop_controlled = x_uncond + w_prop * e_t_prop + u_sw_prop
                else:
                    x_prop_controlled = x_uncond + w_prop * e_t_prop
            else:
                x_prop_controlled = x_uncond + w_prop * e_t_prop
            e_prev_prop = e_t_prop.detach().clone()
            x0_pred = x_uncond + (x_geo_controlled - x_uncond) + (x_prop_controlled - x_uncond)
        else:
            x0_pred = x_geo_controlled

        # ===== Step 4: 反向扩散 =====
        if time[0] < TRUNCATED_TIME:
            x0_pred.sign_()
        x0_pred.clamp_(-1, 1)
        pred_noise = (x_t - alpha * x0_pred) / sigma.clamp(min=1e-8)
        x_t = alpha_next * x0_pred + sigma_next * pred_noise
        x_self_cond = x0_pred
    return x_t

# ===================== 主生成入口 =====================
def generate_validation_set(
    ckpt_path: str,
    output_root: str = "./val_results",
    use_ema: bool = True,
    steps: int = 50,
    truncated_time: float = 0.1,
    cond_w: float = None,
    w_geo: float = 1.5,
    w_prop: float = 1.5,
    use_postprocess: bool = True,
    batch_size: int = 8,
    seed: int = 42,
    num_per_cond: int = 1,
    dataset_folder: str = "",
    save_real_voxels: bool = True,
    max_val_samples: int = 200,
    geo_cond_file: str = "geo_conditions_onehot.npy",
    prop_cond_file: str = "properties.csv",
    # ========== SMC-CFG 参数 ==========
    smc_cfg_enable: bool = False,
    smc_cfg_lambda: float = 5.0,
    smc_cfg_K: float = 0.2,
    no_cfg_warmup_steps: int = 2,
    smc_use_saturation: bool = False,
    smc_saturation_threshold: float = 0.1,
    smc_use_both: bool = False,
    smc_adaptive_K: bool = False,
):
    if cond_w is not None:
        w_geo = cond_w
        w_prop = cond_w

    # 1. 加载模型
    print("▶ 加载模型...")
    from network.model_trainer import DiffusionModel
    discrete_diffusion = DiffusionModel.load_from_checkpoint(ckpt_path).cuda()
    generator = discrete_diffusion.ema_model if use_ema else discrete_diffusion.model
    generator.eval()

    hparams = discrete_diffusion.hparams
    if not dataset_folder:
        dataset_folder = hparams.dataset_folder
    val_ratio = hparams.val_ratio
    use_tensor_condition = getattr(hparams, "use_tensor_condition", False)
    voxel_size = generator.image_size
    NUM_CLASSES = 294
    SAMPLES_PER_CLASS = 100
    TOTAL = NUM_CLASSES * SAMPLES_PER_CLASS

    # 2. 加载数据
    voxel_path = os.path.join(dataset_folder, "voxels.npy")
    all_voxels = np.load(voxel_path).astype(np.float32)
    geo_full_path = os.path.join(dataset_folder, geo_cond_file)
    all_geo_conds = np.load(geo_full_path).astype(np.float32)
    
    all_labels = np.zeros((TOTAL, NUM_CLASSES), dtype=np.float32)
    for cls in range(NUM_CLASSES):
        start = cls * SAMPLES_PER_CLASS
        all_labels[start:start+SAMPLES_PER_CLASS, cls] = 1.0

    all_prop = None
    if use_tensor_condition:
        print("1")
        import pandas as pd
        prop_csv = os.path.join(dataset_folder, prop_cond_file)
        df = pd.read_csv(prop_csv)
        E = df["E"].values
        ANI = df["ANI"].values
        poisson = df["poisson"].values
        prop_raw = np.stack([E, ANI, poisson], axis=-1).astype(np.float32)
        p_min = prop_raw.min(axis=0, keepdims=True)
        p_max = prop_raw.max(axis=0, keepdims=True)
        all_prop = 0.1 + 0.9 * (prop_raw - p_min) / (p_max - p_min + 1e-8)

    _, val_indices = split_random_indices(TOTAL, val_ratio, seed)
    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]

    val_ds = VoxelValDataset(all_voxels, all_geo_conds, all_labels, all_prop, val_indices)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_count = len(val_ds)
    print(f"Validation conditions: {val_count}, samples per condition: {num_per_cond}")

    # 输出目录
    exp_name = Path(ckpt_path).stem
    tag = f"wg{w_geo}_wp{w_prop}"
    
    if smc_cfg_enable:
        tag += f"_SMC_l{smc_cfg_lambda}_K{smc_cfg_K}_warmup{no_cfg_warmup_steps}"
        if smc_use_saturation:
            tag += "_sat"
        if smc_use_both:
            tag += "_both"
        if smc_adaptive_K:
            tag += "_adaptive"
    else:
        tag += "_CFG"
    
    if use_postprocess:
        tag += "_pp"
    
    out_dir = Path(output_root) / f"{exp_name}_{tag}"
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    real_dir = out_dir / "real"
    if save_real_voxels:
        real_dir.mkdir(parents=True, exist_ok=True)

    # 批量生成
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch_geo = batch["condition"].cuda()
            batch_prop = batch["tensor_feature"].cuda() if use_tensor_condition else None
            bsz = batch_geo.shape[0]
            geo_rep = batch_geo.repeat_interleave(num_per_cond, dim=0)
            prop_rep = batch_prop.repeat_interleave(num_per_cond, dim=0) if batch_prop is not None else None
            cur_bs = geo_rep.shape[0]

            if smc_cfg_enable:
                # 直接调用SMC-CFG纯函数
                res = sample_smc_cfg(
                    generator=generator,
                    geo_condition=geo_rep,
                    prop=prop_rep,
                    batch_size=cur_bs,
                    steps=steps,
                    truncated_time=truncated_time,
                    cond_w=cond_w,
                    w_geo=w_geo,
                    w_prop=w_prop,
                    smc_cfg_enable=True,
                    smc_cfg_lambda=smc_cfg_lambda,
                    smc_cfg_K=smc_cfg_K,
                    no_cfg_warmup_steps=no_cfg_warmup_steps,
                    use_saturation=smc_use_saturation,
                    saturation_threshold=smc_saturation_threshold,
                    use_smc_for_prop=smc_use_both,
                    adaptive_K=smc_adaptive_K,
                    verbose=False,
                )
            else:
                # 标准CFG
                res = generator.sample_with_all_condition(
                    geo_condition=geo_rep,
                    prop=prop_rep,
                    batch_size=cur_bs,
                    steps=steps,
                    truncated_index=truncated_time,
                    cond_w=cond_w,
                    w_geo=w_geo,
                    w_prop=w_prop,
                    use_vf_constraint=False,
                    verbose=False,
                )

            # 保存
            for bid in range(bsz):
                real_vox = all_voxels[val_indices[global_idx]]
                if save_real_voxels:
                    save_voxel((real_vox>0.5).astype(bool), str(real_dir / f"{global_idx}.voxel"))
                for k in range(num_per_cond):
                    pos = bid * num_per_cond + k
                    vox_np = res[pos,0].cpu().numpy()
                    bin_vox = (vox_np > 0).astype(bool)
                    if use_postprocess:
                        bin_vox = voxel_postprocess(bin_vox)
                    save_voxel(bin_vox, str(gen_dir / f"{global_idx}_{k}.voxel"))
                global_idx += 1
    
    print(f"Generation complete. Output: {out_dir}")

@torch.no_grad()
def generate_visual_samples(
    generator,
    geo_conditions: np.ndarray,
    prop_conditions: np.ndarray = None,
    num_per_condition: int = 1,
    batch_size: int = 8,
    steps: int = 50,
    truncated_time: float = 0.1,
    w_geo: float = 1.5,
    w_prop: float = 1.5,
    use_postprocess: bool = True,
    # SMC-CFG参数
    smc_cfg_enable: bool = False,
    smc_cfg_lambda: float = 5.0,
    smc_cfg_K: float = 0.2,
    no_cfg_warmup_steps: int = 2,
    use_saturation: bool = False,
    saturation_threshold: float = 0.1,
    use_smc_for_prop: bool = False,
    adaptive_K: bool = False,
    save_dir: str = None,
    verbose: bool = True
) -> np.ndarray:
    """
    【可视化专用生成函数】
    输入自定义一批条件，每个条件生成num_per_condition个样本，返回全部体素
    Args:
        generator: 模型ema_model
        geo_conditions: np.ndarray, [N, geo_dim]  N组几何条件
        prop_conditions: np.ndarray, [N, prop_dim]  N组力学条件，None则不使用属性条件
        num_per_condition: 每个条件生成多少个样本
        batch_size:推理batch
        save_dir: 如果提供路径，会保存.voxel文件
    Returns:
        all_voxels_np: array [N_total, H,W,Z]  bool体素（经过后处理）
    """
    device = generator.device
    N_cond = geo_conditions.shape[0]
    # repeat条件：每个条件复制num_per_condition次
    geo_cond_np = np.repeat(geo_conditions, num_per_condition, axis=0)
    use_tensor_condition = getattr(generator, "use_tensor_condition", False)
    if use_tensor_condition and prop_conditions is not None:
        prop_cond_np = np.repeat(prop_conditions, num_per_condition, axis=0)
    else:
        prop_cond_np = None

    all_results = []
    total_samples = geo_cond_np.shape[0]
    indices = np.arange(total_samples)

    for start in tqdm(range(0, total_samples, batch_size), desc="Visual sample generation", disable=not verbose):
        end = min(start + batch_size, total_samples)
        batch_idx = indices[start:end]
        geo_batch = torch.from_numpy(geo_cond_np[batch_idx]).float().to(device)
        if prop_cond_np is not None:
            prop_batch = torch.from_numpy(prop_cond_np[batch_idx]).float().to(device)
        else:
            prop_batch = None

        cur_bs = geo_batch.shape[0]
        if smc_cfg_enable:
            gen_out = sample_smc_cfg(
                generator=generator,
                geo_condition=geo_batch,
                prop=prop_batch,
                batch_size=cur_bs,
                steps=steps,
                truncated_time=truncated_time,
                w_geo=w_geo,
                w_prop=w_prop,
                smc_cfg_enable=True,
                smc_cfg_lambda=smc_cfg_lambda,
                smc_cfg_K=smc_cfg_K,
                no_cfg_warmup_steps=no_cfg_warmup_steps,
                use_saturation=use_saturation,
                saturation_threshold=saturation_threshold,
                use_smc_for_prop=use_smc_for_prop,
                adaptive_K=adaptive_K,
                verbose=False
            )
        else:
            gen_out = generator.sample_with_all_condition(
                geo_condition=geo_batch,
                prop=prop_batch,
                batch_size=cur_bs,
                steps=steps,
                truncated_index=truncated_time,
                w_geo=w_geo,
                w_prop=w_prop,
                use_vf_constraint=False,
                verbose=False
            )
        # 转numpy + 二值
        vox_batch = gen_out[:,0].cpu().numpy()
        bin_batch = (vox_batch > 0).astype(bool)
        # 后处理
        if use_postprocess:
            for i in range(bin_batch.shape[0]):
                bin_batch[i] = voxel_postprocess(bin_batch[i])
        all_results.append(bin_batch)

    all_voxels_np = np.concatenate(all_results, axis=0)
    # 可选保存
    if save_dir is not None:
        save_p = Path(save_dir)
        save_p.mkdir(parents=True, exist_ok=True)
        cnt = 0
        for cond_id in range(N_cond):
            for k in range(num_per_condition):
                vox = all_voxels_np[cnt]
                save_voxel(vox, str(save_p / f"cond{cond_id}_sample{k}.voxel"))
                cnt +=1
    return all_voxels_np


@torch.no_grad()
def generate_random_10cond_3samples(
    ckpt_path: str,
    dataset_folder: str,
    out_root: str = "./rand10_vis_output",
    seed=42,
    max_val_samples=200
):
    """
    从验证集随机抽取10个条件，每个条件生成3个样本，固定ST-Control参数
    """
    # fixed params
    w_geo = 2.0
    w_prop = 1.6
    steps = 100
    use_postprocess = False
    smc_cfg_enable = True
    smc_cfg_lambda = 5.0
    smc_cfg_K = 0.2
    no_cfg_warmup_steps = 2
    use_saturation = True
    use_smc_for_prop = True
    adaptive_K = False
    

    # load model
    from network.model_trainer import DiffusionModel
    discrete_diffusion = DiffusionModel.load_from_checkpoint(ckpt_path).cuda()
    generator = discrete_diffusion.ema_model
    generator.eval()
    hparams = discrete_diffusion.hparams
    val_ratio = hparams.val_ratio
    use_tensor_condition = getattr(hparams, "use_tensor_condition", False)

    # load full dataset
    NUM_CLASSES = 294
    SAMPLES_PER_CLASS = 100
    TOTAL = NUM_CLASSES * SAMPLES_PER_CLASS
    all_voxels = np.load(os.path.join(dataset_folder, "voxels.npy")).astype(np.float32)
    all_geo_conds = np.load(os.path.join(dataset_folder, "geo_conditions_onehot.npy")).astype(np.float32)

    all_labels = np.zeros((TOTAL, NUM_CLASSES), dtype=np.float32)
    for cls in range(NUM_CLASSES):
        start = cls * SAMPLES_PER_CLASS
        all_labels[start:start+SAMPLES_PER_CLASS, cls] = 1.0

    all_prop = None
    if use_tensor_condition:
        import pandas as pd
        df = pd.read_csv(os.path.join(dataset_folder, "properties.csv"))
        E = df["E"].values
        ANI = df["ANI"].values
        poisson = df["poisson"].values
        prop_raw = np.stack([E, ANI, poisson], axis=-1).astype(np.float32)
        p_min = prop_raw.min(axis=0, keepdims=True)
        p_max = prop_raw.max(axis=0, keepdims=True)
        all_prop = 0.1 + 0.9 * (prop_raw - p_min) / (p_max - p_min + 1e-8)

    # get val indices
    _, val_indices = split_random_indices(TOTAL, val_ratio, seed=seed)
    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]

    # randomly sample 10 unique condition indices from validation set
    rng = np.random.default_rng(seed=seed+1)
    selected_10_idx = rng.choice(val_indices, size=10, replace=False)
    print(f"Selected 10 sample indices from val set: {selected_10_idx}")

    geo_10 = all_geo_conds[selected_10_idx]
    if use_tensor_condition:
        prop_10 = all_prop[selected_10_idx]
    else:
        prop_10 = None

    # call visual sample generation
    vis_voxels = generate_visual_samples(
        generator=generator,
        geo_conditions=geo_10,
        prop_conditions=prop_10,
        num_per_condition=3,
        batch_size=8,
        steps=steps,
        truncated_time=0.1,
        w_geo=w_geo,
        w_prop=w_prop,
        use_postprocess=use_postprocess,
        smc_cfg_enable=smc_cfg_enable,
        smc_cfg_lambda=smc_cfg_lambda,
        smc_cfg_K=smc_cfg_K,
        no_cfg_warmup_steps=no_cfg_warmup_steps,
        use_saturation=use_saturation,
        saturation_threshold=0.1,
        use_smc_for_prop=use_smc_for_prop,
        adaptive_K=adaptive_K,
        save_dir=out_root,
        verbose=True
    )
    print(f"Finished! Generated shape = {vis_voxels.shape}")
    print(f"Saved samples to {out_root}")
    return vis_voxels, selected_10_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default="./val_results")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--w_geo", type=float, default=2.0)
    parser.add_argument("--w_prop", type=float, default=1.6)
    parser.add_argument("--num_per", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_val", type=int, default=200)
    parser.add_argument("--no_pp", action="store_true")
    
    parser.add_argument("--smc_enable", action="store_true")
    parser.add_argument("--smc_lambda", type=float, default=5.0)
    parser.add_argument("--smc_K", type=float, default=0.2)
    parser.add_argument("--smc_warmup", type=int, default=2)
    parser.add_argument("--smc_sat", action="store_true")
    parser.add_argument("--smc_sat_thresh", type=float, default=0.1)
    parser.add_argument("--smc_both", action="store_true")
    parser.add_argument("--smc_adaptive", action="store_true")
    parser.add_argument("--rand10vis", action="store_true", help="Randomly pick 10 conditions from val set, each generate 3 samples, fixed STC params")

    args = parser.parse_args()
    if args.rand10vis:
        generate_random_10cond_3samples(
            ckpt_path=args.ckpt,
            dataset_folder="/home/daibingxuan/workspace/metaset/3D_diverse/2020_JMD/shape_generation_code/metaset_64_npy",
            out_root="./rand10_vis_output",
            seed=23,
            max_val_samples=args.max_val
        )
    else:
        generate_validation_set(
            ckpt_path=args.ckpt,
            output_root=args.output,
            steps=args.steps,
            w_geo=args.w_geo,
            w_prop=args.w_prop,
            use_postprocess=not args.no_pp,
            num_per_cond=args.num_per,
            batch_size=args.batch,
            max_val_samples=args.max_val,
            smc_cfg_enable=args.smc_enable,
            smc_cfg_lambda=args.smc_lambda,
            smc_cfg_K=args.smc_K,
            no_cfg_warmup_steps=args.smc_warmup,
            smc_use_saturation=args.smc_sat,
            smc_saturation_threshold=args.smc_sat_thresh,
            smc_use_both=args.smc_both,
            smc_adaptive_K=args.smc_adaptive,
        )