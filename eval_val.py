import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label
from bitstring import BitArray
import argparse
from collections import defaultdict
from itertools import combinations


# ==============================================================================
# 1. 工具函数：加载生成的.voxel文件
# ==============================================================================
def load_gen_voxel(voxel_path: str, dim=(64, 64, 64)):
    expected_bits = dim[0] * dim[1] * dim[2]
    try:
        with open(voxel_path, 'rb') as f:
            bits = BitArray(f.read())
        
        if len(bits) != expected_bits:
            print(f"⚠️  文件位数不匹配，跳过: {voxel_path}")
            return None
        
        reordered = BitArray()
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            reordered.append(byte[::-1])
        
        voxel = np.array(list(reordered), dtype=np.bool_)
        voxel = voxel.reshape(dim, order='C')
        return voxel
    except Exception as e:
        print(f"⚠️  加载失败，跳过: {voxel_path}, 错误: {str(e)}")
        return None


# ==============================================================================
# 2. 单样本指标计算
# ==============================================================================
def calc_volume_fraction(voxel):
    return float(voxel.mean())

def calc_connectivity(voxel):
    struct = np.ones((3, 3, 3), dtype=bool)
    _, num_features = label(voxel, structure=struct)
    return num_features == 1, num_features

def calc_dangling(voxel, min_size=10):
    struct = np.ones((3, 3, 3), dtype=bool)
    labeled, num_features = label(voxel, structure=struct)
    if num_features <= 1:
        return True, 0
    domain_sizes = np.bincount(labeled.flat)[1:]
    small_count = int(np.sum(domain_sizes < min_size))
    return small_count == 0, small_count

def calc_symmetry(voxel):
    iou_list = []
    for axes in [(1, 2), (0, 2), (0, 1)]:
        rot = np.rot90(voxel, k=1, axes=axes)
        inter = np.logical_and(voxel, rot).sum()
        union = np.logical_or(voxel, rot).sum()
        iou_list.append(inter / union if union > 0 else 0)
    inv = voxel[::-1, ::-1, ::-1]
    inter = np.logical_and(voxel, inv).sum()
    union = np.logical_or(voxel, inv).sum()
    iou_list.append(inter / union if union > 0 else 0)
    return float(np.mean(iou_list))

def calc_periodicity(voxel):
    x_match = float(np.mean(voxel[0, :, :] == voxel[-1, :, :]))
    y_match = float(np.mean(voxel[:, 0, :] == voxel[:, -1, :]))
    z_match = float(np.mean(voxel[:, :, 0] == voxel[:, :, -1]))
    avg = (x_match + y_match + z_match) / 3
    return avg >= 0.98, avg

def calc_voxel_iou(v1, v2):
    inter = np.logical_and(v1, v2).sum()
    union = np.logical_or(v1, v2).sum()
    return inter / union if union > 0 else 0.0


# ==============================================================================
# 3. 主评估函数
# ==============================================================================
def evaluate_validation(
    result_dir: str,
    dataset_folder: str,
    dangling_min_size: int = 10,
    periodic_threshold: float = 0.98,
    coverage_iou_threshold: float = 0.8,
    save_per_sample_error: bool = True
):
    gen_dir = os.path.join(result_dir, "generated")

    # -------------------- 1. 加载原始真实体素数据集 --------------------
    print("▶ 加载原始真实体素数据集...")
    voxels_path = os.path.join(dataset_folder, "voxels.npy")
    indices_path = os.path.join("/home/daibingxuan/workspace/microstructure_generation_3d/val_results/best-val_loss-epoch=181-val_loss=0.0126_n3_w1.8", "val_indices.npy")
    
    if not os.path.exists(voxels_path):
        raise FileNotFoundError(f"找不到原始体素文件: {voxels_path}")
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"找不到验证集索引文件: {indices_path}")
    
    real_voxels_all = np.load(voxels_path, mmap_mode='r')
    val_indices = np.load(indices_path)
    print(f"  原始数据集总样本数: {len(real_voxels_all)}")
    print(f"  验证集样本数: {len(val_indices)}")

    # -------------------- 2. 解析生成文件，按条件ID分组 --------------------
    gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith('.voxel')])
    if len(gen_files) == 0:
        raise ValueError(f"生成目录 {gen_dir} 下没有找到 .voxel 文件")

    cond_groups = defaultdict(list)
    for fname in gen_files:
        name = fname.replace('.voxel', '')
        cond_id = name.split('_')[0] if '_' in name else name
        cond_groups[cond_id].append(fname)

    cond_ids = sorted(cond_groups.keys(), key=lambda x: int(x))
    num_conds = len(cond_ids)
    num_per_cond = len(cond_groups[cond_ids[0]])

    print("=" * 70)
    print(f"📂 结果目录: {result_dir}")
    print(f"  条件总数: {num_conds}")
    print(f"  每条件生成样本数: {num_per_cond}")
    print(f"  总生成样本数: {len(gen_files)}")
    print("=" * 70)

    # -------------------- 3. 初始化统计容器 --------------------
    # 条件级统计
    cond_stats = {
        'vf_mean': [],
        'conn_valid_rate': [],
        'dang_valid_rate': [],
        'sym_mean': [],
        'per_mean': [],
        'per_valid_rate': [],
        'vf_mae': [],
        'iou_mean': [],
    }
    real_vf_list = []

    # 逐样本统计（所有生成样本平铺）
    per_sample = {
        'vf': [],
        'vf_error': [],
        'conn_valid': [],
        'dang_valid': [],
        'sym': [],
        'per': [],
        'iou': [],
    }

    # 多样性统计
    diversity_stats = {
        'intra_div': [],
        'coverage_flags': [],
    }

    skip_count = 0

    # -------------------- 4. 逐条件遍历计算 --------------------
    for cond_id in tqdm(cond_ids, desc="评估进度"):
        gen_fnames = cond_groups[cond_id]
        
        # 取出对应真实体素与真实体积分数
        original_idx = val_indices[int(cond_id)]
        real_voxel = real_voxels_all[original_idx].astype(np.bool_)
        real_vf = calc_volume_fraction(real_voxel)
        real_vf_list.append(real_vf)

        # 组内临时变量
        group_vf = []
        group_conn_valid = []
        group_dang_valid = []
        group_sym = []
        group_per = []
        group_per_valid = []
        group_iou = []
        group_vf_err = []
        group_gen_voxels = []

        for fname in gen_fnames:
            gen_path = os.path.join(gen_dir, fname)
            gen_voxel = load_gen_voxel(gen_path)
            if gen_voxel is None:
                skip_count += 1
                continue

            # 计算单样本指标
            vf = calc_volume_fraction(gen_voxel)
            conn_valid, _ = calc_connectivity(gen_voxel)
            dang_valid, _ = calc_dangling(gen_voxel, min_size=dangling_min_size)
            sym = calc_symmetry(gen_voxel)
            per_valid, per_score = calc_periodicity(gen_voxel)
            iou = calc_voxel_iou(gen_voxel, real_voxel)
            vf_err = abs(vf - real_vf)  # 逐样本体积分数绝对误差

            # 组内收集
            group_vf.append(vf)
            group_conn_valid.append(conn_valid)
            group_dang_valid.append(dang_valid)
            group_sym.append(sym)
            group_per.append(per_score)
            group_per_valid.append(per_valid)
            group_iou.append(iou)
            group_vf_err.append(vf_err)
            group_gen_voxels.append(gen_voxel)

            # 逐样本全局收集
            per_sample['vf'].append(vf)
            per_sample['vf_error'].append(vf_err)
            per_sample['conn_valid'].append(conn_valid)
            per_sample['dang_valid'].append(dang_valid)
            per_sample['sym'].append(sym)
            per_sample['per'].append(per_score)
            per_sample['iou'].append(iou)

        # 条件级聚合
        if len(group_vf) > 0:
            cond_stats['vf_mean'].append(np.mean(group_vf))
            cond_stats['conn_valid_rate'].append(np.mean(group_conn_valid))
            cond_stats['dang_valid_rate'].append(np.mean(group_dang_valid))
            cond_stats['sym_mean'].append(np.mean(group_sym))
            cond_stats['per_mean'].append(np.mean(group_per))
            cond_stats['per_valid_rate'].append(np.mean(group_per_valid))
            cond_stats['vf_mae'].append(np.mean(group_vf_err))
            cond_stats['iou_mean'].append(np.mean(group_iou))

            # 多样性计算
            pair_ious = []
            for a, b in combinations(group_gen_voxels, 2):
                pair_ious.append(calc_voxel_iou(a, b))
            intra_div = 1.0 - np.mean(pair_ious) if len(pair_ious) > 0 else 0.0
            diversity_stats['intra_div'].append(intra_div)
            diversity_stats['coverage_flags'].append(np.max(group_iou) >= coverage_iou_threshold)

    # -------------------- 5. 保存逐样本误差 --------------------
    if save_per_sample_error:
        save_path = os.path.join(result_dir, "vf_errors_per_sample.npy")
        np.save(save_path, np.array(per_sample['vf_error']))
        print(f"\n💾 逐样本体积分数误差已保存至: {save_path}")

    # -------------------- 6. 结果汇总输出 --------------------
    print("\n" + "=" * 70)
    print("📊 评估结果汇总")
    if skip_count > 0:
        print(f"⚠️  共跳过 {skip_count} 个损坏/无法读取的生成文件")
    print("=" * 70)

    print("\n━━━━━━━━━━ 【条件级指标】 ━━━━━━━━")
    print(f"\n📐 体积分数控制精度")
    vf_means = np.array(cond_stats['vf_mean'])
    real_vf_arr = np.array(real_vf_list)
    vf_maes_cond = np.array(cond_stats['vf_mae'])
    print(f"  生成均值: {vf_means.mean():.4f} ± {vf_means.std():.4f}")
    print(f"  真实均值: {real_vf_arr.mean():.4f} ± {real_vf_arr.std():.4f}")
    print(f"  条件级平均MAE: {vf_maes_cond.mean():.4f}（先条件内平均，再跨条件平均）")

    print(f"\n🔗 连通性有效性 V_C")
    print(f"  条件平均单连通率: {np.mean(cond_stats['conn_valid_rate']):.4f}")

    print(f"\n🧩 悬空节点约束 V_DR")
    print(f"  条件平均无缺陷率: {np.mean(cond_stats['dang_valid_rate']):.4f}")

    print(f"\n🔄 对称性得分 V_S")
    sym_means = np.array(cond_stats['sym_mean'])
    print(f"  条件平均得分: {sym_means.mean():.4f} ± {sym_means.std():.4f}")
    print(f"  得分>0.95占比: {np.mean(sym_means > 0.95):.4f}")

    print(f"\n📦 周期性一致性 V_P")
    per_means = np.array(cond_stats['per_mean'])
    print(f"  条件平均匹配率: {per_means.mean():.4f} ± {per_means.std():.4f}")
    print(f"  条件平均达标率: {np.mean(cond_stats['per_valid_rate']):.4f}")

    print(f"\n🎯 结构相似度 IoU（生成 vs 真实）")
    iou_means = np.array(cond_stats['iou_mean'])
    print(f"  条件平均IoU: {iou_means.mean():.4f} ± {iou_means.std():.4f}")

    print("\n━━━━━━━━━━ 【逐样本级指标】（逐样本计算后取平均） ━━━━━━━━")
    vf_err_arr = np.array(per_sample['vf_error'])
    print(f"\n📐 体积分数误差（逐样本直接平均）")
    print(f"  逐样本平均MAE: {vf_err_arr.mean():.4f}")
    print(f"  误差标准差: {vf_err_arr.std():.4f}")
    print(f"  最小误差: {vf_err_arr.min():.4f}")
    print(f"  最大误差: {vf_err_arr.max():.4f}")
    print(f"  误差中位数: {np.median(vf_err_arr):.4f}")

    print(f"\n🔗 单连通占比: {np.mean(per_sample['conn_valid']):.4f}")
    print(f"🧩 无悬空占比: {np.mean(per_sample['dang_valid']):.4f}")
    print(f"🔄 对称得分均值: {np.mean(per_sample['sym']):.4f}")
    print(f"📦 周期匹配率均值: {np.mean(per_sample['per']):.4f}")
    print(f"🎯 全体样本IoU均值: {np.mean(per_sample['iou']):.4f}")

    print("\n━━━━━━━━━━ 【多样性指标】 ━━━━━━━━")
    intra_div_arr = np.array(diversity_stats['intra_div'])
    coverage_rate = np.mean(diversity_stats['coverage_flags'])
    print(f"\n🎲 同条件内多样性")
    print(f"  全局平均多样性: {intra_div_arr.mean():.4f} ± {intra_div_arr.std():.4f}")
    print(f"  说明：数值越高，同一条件下生成结果差异越大")

    print(f"\n📊 真实模式覆盖度 (IoU≥{coverage_iou_threshold})")
    print(f"  覆盖条件占比: {coverage_rate:.4f}")

    print("=" * 70)
    return cond_stats, per_sample, diversity_stats


# ==============================================================================
# 4. 命令行入口
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='验证集全指标评估（含逐样本误差）')
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--dataset_folder', type=str, required=True)
    parser.add_argument('--dangling_min_size', type=int, default=10)
    parser.add_argument('--periodic_threshold', type=float, default=0.98)
    parser.add_argument('--coverage_iou_threshold', type=float, default=0.8)
    parser.add_argument('--save_per_sample_error', type=bool, default=True)
    
    args = parser.parse_args()
    evaluate_validation(
        result_dir=args.result_dir,
        dataset_folder=args.dataset_folder,
        dangling_min_size=args.dangling_min_size,
        periodic_threshold=args.periodic_threshold,
        coverage_iou_threshold=args.coverage_iou_threshold,
        save_per_sample_error=args.save_per_sample_error
    )