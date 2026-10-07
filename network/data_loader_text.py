import os
import numpy as np
import pandas as pd
import collections
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy.ndimage import rotate
import random
import joblib

class VoxelDataset(Dataset):
    def __init__(
        self,
        dataset_folder=None,
        transform=None,
        use_tensor_condition=False,
        cond_type="onehot",
        cond_file=None,
    ):
        """
        Args:
            dataset_folder: 数据集根目录（包含 all_voxels.npy、geo_conditions_xxx.npy 等）
            transform: 图像变换（保留兼容）
            use_tensor_condition: 是否启用额外弹性属性条件
            cond_type: 定量几何条件类型
                - "onehot": 高维独热版（推荐扩散模型使用，默认）
                - "compact": 低维整数精简版（适合带Embedding层的模型）
            cond_file: 自定义条件npy文件路径，为None时自动从dataset_folder加载
        """
        self.dataset_path = dataset_folder
        self.transform = transform
        self.use_tensor_condition = use_tensor_condition
        self.cond_type = cond_type

        voxel_path = os.path.join(dataset_folder, "voxels.npy") if dataset_folder else \
            "/home/daibingxuan/workspace/metaset/3D_diverse/2020_JMD/shape_generation_code/metaset_64_npy/voxels.npy"
        self.voxels = np.load(voxel_path)
        self.voxels = self.voxels.astype(np.float32)

        if cond_file is None:
            if cond_type == "onehot":
                cond_file = os.path.join(dataset_folder, "geo_conditions_onehot.npy") if dataset_folder else \
            "/home/daibingxuan/workspace/metaset/3D_diverse/2020_JMD/shape_generation_code/metaset_64_npy/geo_conditions_onehot.npy"
            elif cond_type == "compact":
                cond_file = os.path.join(dataset_folder, "geo_conditions_compact.npy") if dataset_folder else \
            "/home/daibingxuan/workspace/metaset/3D_diverse/2020_JMD/metaset_jmd_294types_100_npy/geo_conditions_compact.npy"
            else:
                raise ValueError(f"不支持的cond_type: {cond_type}，可选 onehot / compact")

        self.geo_conditions = np.load(cond_file).astype(np.float32)
        assert len(self.geo_conditions) == len(self.voxels), \
            "定量条件数量与体素数量不匹配，请检查文件是否对应"

        NUM_CLASSES = 294
        SAMPLES_PER_CLASS = 100
        TOTAL = NUM_CLASSES * SAMPLES_PER_CLASS
        self.labels = np.zeros((TOTAL, NUM_CLASSES), dtype=np.float32)
        for cls in range(NUM_CLASSES):
            start = cls * SAMPLES_PER_CLASS
            self.labels[start:start+SAMPLES_PER_CLASS, cls] = 1.0

        # if self.use_tensor_condition:
        #     properties_file = os.path.join(dataset_folder, "properties.csv")
        #     properties = pd.read_csv(properties_file)
        #     E_label = np.expand_dims(1 / properties["E"].values, axis=-1)
        #     Ani_label = np.expand_dims(properties["Anisotropy"].values, axis=-1)
        #     phi = np.expand_dims(properties["Phi"].values, axis=-1)
        #     self.prop = np.concatenate((phi, E_label, Ani_label), axis=-1).astype(np.float32)
        #     assert len(self.prop) == len(self.voxels), "弹性属性数量与体素不匹配"
        if self.use_tensor_condition:
            properties_file = os.path.join(dataset_folder, "properties.csv")
            properties = pd.read_csv(properties_file)
            
            # # 读取刚度矩阵分量
            # C11 = properties["C11"].values
            # C12 = properties["C12"].values
            # C44 = properties["C44"].values
            
            # # 1. 杨氏模量 E（沿[100]晶向）
            # E = (C11 - C12) * (C11 + 2 * C12) / (C11 + C12)
            # # 2. Zener各向异性指数 ANI
            # ANI = 2 * C44 / (C11 - C12)
            # # 3. 泊松比 ν
            # poisson = C12 / (C11 + C12)
            E = properties["E"].values
            ANI = properties["ANI"].values
            poisson = properties["poisson"].values
            
            # 扩维后拼接为 [E, ANI, 泊松比] 3维属性向量
            E_label = np.expand_dims(E, axis=-1)
            ANI_label = np.expand_dims(ANI, axis=-1)
            poisson_label = np.expand_dims(poisson, axis=-1)
            
            self.prop = np.concatenate((E_label, ANI_label, poisson_label), axis=-1).astype(np.float32)
            # ========== Min-Max 归一化到 [0.1, 1.0]，0 保留为缺失标识 ==========
            self.prop_min = self.prop.min(axis=0, keepdims=True)
            self.prop_max = self.prop.max(axis=0, keepdims=True)
            self.prop = 0.1 + 0.9 * (self.prop - self.prop_min) / (self.prop_max - self.prop_min + 1e-8)
            self.prop = self.prop.astype(np.float32)
            assert len(self.prop) == len(self.voxels), "弹性属性数量与体素不匹配"
        else:
            self.prop = None

    def __len__(self):
        return len(self.voxels)

    def augment_voxel_tensor(self, voxel, p_rotate=0.5, p_flip=0.5, p_dropout=0.2):
        """
        voxel: torch tensor of shape (1, D, H, W), binary (0/1)
        returns: augmented voxel of the same shape
        """
        voxel_np = voxel.squeeze(0).numpy()

        # 1. 随机旋转
        if random.random() < p_rotate:
            axes = random.choice([(0, 1), (1, 2), (0, 2)])
            angle = random.choice([90, 180, 270])
            voxel_np = rotate(voxel_np, angle=angle, axes=axes, reshape=False, order=0, mode='nearest')

        # 2. 随机翻转
        if random.random() < p_flip:
            axis = random.choice([0, 1, 2])
            voxel_np = np.flip(voxel_np, axis=axis)

        # 3. dropout
        if random.random() < p_dropout:
            dropout_mask = np.random.binomial(1, 0.95, voxel_np.shape)
            voxel_np = voxel_np * dropout_mask

        voxel_np = (voxel_np > 0.5).astype(np.float32)
        return torch.from_numpy(voxel_np).unsqueeze(0)

    def __getitem__(self, idx):
        res = {}
        voxel = torch.tensor(self.voxels[idx])
        # voxel = self.augment_voxel_tensor(voxel)  # 按需开启增强
        voxel = 2 * voxel - 1  # [0,1] → [-1,1]
        voxel = voxel.unsqueeze(0)
        res["occupancy"] = voxel

        geo_cond = torch.tensor(self.geo_conditions[idx], dtype=torch.float32)
        res["condition"] = geo_cond

        res["label"] = torch.tensor(self.labels[idx])
        res["idx"] = idx
        # print("11111")
        if self.use_tensor_condition:
            prop = torch.tensor(self.prop[idx], dtype=torch.float32)
            res["tensor_feature"] = prop
        else:
            res["tensor_feature"] = torch.zeros(0)

        return res
