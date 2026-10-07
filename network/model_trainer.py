import copy
from utils.utils import set_requires_grad
from torch.utils.data import DataLoader, ConcatDataset, Subset
from network.model_utils import EMA
from network.data_loader_text import VoxelDataset, VoxelDataset1
from pathlib import Path
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import update_moving_average
from pytorch_lightning import LightningModule
from network.model import OccupancyDiffusion
import torch
import torch.nn as nn
import os
import random


def split_random(dataset, val_ratio=0.2, seed=42):
    """
    全样本随机打乱划分训练/验证集
    Args:
        dataset: 原始数据集
        val_ratio: 验证集占比，默认0.2（20%）
        seed: 随机种子，保证划分可复现
    """
    random.seed(seed)
    total_samples = len(dataset)
    indices = list(range(total_samples))
    random.shuffle(indices)

    split_point = int(total_samples * (1 - val_ratio))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(f"==== 随机划分数据集 ====")
    print(f"总样本数: {total_samples}")
    print(f"训练集: {len(train_indices)} 样本 ({(1-val_ratio)*100:.0f}%)")
    print(f"验证集: {len(val_indices)} 样本 ({val_ratio*100:.0f}%)")
    return train_set, val_set


class DiffusionModel(LightningModule):
    def __init__(
        self,
        dataset_folder: str = "",
        results_folder: str = './results',
        voxel_folder: str = "",
        csv_path: str = "",
        image_size: int = 32,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        attention_resolutions: str = "16,8",
        optimizier: str = "adam",
        with_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        ema_rate: float = 0.999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        use_geo_condition: bool = True,
        cond_type: str = "onehot",
        geo_condition_dim: int = 50,
        use_tensor_condition: bool = False,
        tensor_condition_dim: int = 3,
        noise_schedule: str = "linear",
        debug: bool = False,
        # 条件缺失配置（几何分块缺失 + 属性单维度缺失）
        geo_cond_dropout: float = 0.1,          # 几何条件整体缺失概率
        geo_block_dropout: float = 0.15,        # 几何条件每个语义块独立缺失概率
        tensor_global_dropout: float = 0.1,     # 属性条件整体缺失概率
        tensor_per_dim_dropout: float = 0.15,   # 属性单维度缺失概率（整体保留时生效）
        # 验证集配置
        use_validation: bool = True,
        val_batch_size: int = 8,
        val_ratio: float = 0.2,  # 验证集占比
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.results_folder = Path(results_folder)

        self.model = OccupancyDiffusion(
            image_size=image_size,
            base_channels=base_channels,
            attention_resolutions=attention_resolutions,
            with_attention=with_attention,
            dropout=dropout,
            use_geo_condition=use_geo_condition,
            geo_condition_dim=geo_condition_dim,
            use_tensor_condition=use_tensor_condition,
            tensor_condition_dim=tensor_condition_dim,
            num_heads=num_heads,
            noise_schedule=noise_schedule,
            verbose=verbose
        )

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.lr = lr
        self.image_size = image_size
        self.dataset_folder = dataset_folder
        self.voxel_folder = voxel_folder
        self.csv_path = csv_path
        self.with_attention = with_attention
        self.save_every_epoch = save_every_epoch
        self.training_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.use_geo_condition = use_geo_condition
        self.cond_type = cond_type
        self.use_tensor_condition = use_tensor_condition
        # 保存条件缺失参数
        self.geo_cond_dropout = geo_cond_dropout
        self.geo_block_dropout = geo_block_dropout
        self.tensor_global_dropout = tensor_global_dropout
        self.tensor_per_dim_dropout = tensor_per_dim_dropout
        self.use_validation = use_validation
        self.val_ratio = val_ratio

        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.optimizier = optimizier
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)

        self.num_workers = 0 if debug else 2
        # 提前初始化数据集并计算迭代次数，避免sanity check报错
        self._init_datasets()

    def _init_datasets(self):
        _dataset = VoxelDataset(
            dataset_folder=self.dataset_folder,
            cond_type=self.cond_type,
            use_tensor_condition=self.use_tensor_condition
        )

        if self.use_validation:
            self.train_dataset, self.val_dataset = split_random(
                _dataset, val_ratio=self.val_ratio
            )
        else:
            self.train_dataset = _dataset
            self.val_dataset = None

        # 提前计算训练迭代总步数（向上取整，与dataloader长度一致）
        self.iterations = (len(self.train_dataset) + self.batch_size - 1) // self.batch_size

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        if self.optimizier == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizier == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=5, min_lr=1e-5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" if self.use_validation else "loss",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )
        # 再次赋值确保与实际dataloader一致，不影响原有逻辑
        self.iterations = len(dataloader)
        return dataloader

    def val_dataloader(self):
        if not self.use_validation or self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

    def training_step(self, batch, batch_idx):
        occupancy = batch["occupancy"]
        geo_condition = batch["condition"]
        tensor_feature = batch["tensor_feature"] if self.use_tensor_condition else None

        # ========== 条件缺失逻辑 ==========
        batch_size = occupancy.shape[0]
        device = occupancy.device

        # 1. 几何条件：双层缺失（整体缺失 + 语义分块独立缺失）
        if self.use_geo_condition:
            # 几何条件语义分块定义（基于50维标准构成）
            # [0:34]空间群独热、[34:37]结构类型独热、[37:41]拓扑变体独热、[41:50]连续几何参数
            geo_blocks = [
                (0, 34),
                (34, 37),
                (37, 41),
                (41, 50)
            ]
            
            # 初始化全1掩码
            geo_mask = torch.ones(batch_size, geo_condition.shape[1], device=device)
            
            # 第一层：整体缺失（整个几何条件全部丢弃，对应纯无条件）
            if self.geo_cond_dropout > 0:
                global_geo_mask = torch.rand(batch_size, 1, device=device) > self.geo_cond_dropout
                geo_mask = geo_mask * global_geo_mask.float()
            
            # 第二层：分块独立缺失（每个语义块单独随机丢弃，整体保留时生效）
            if self.geo_block_dropout > 0:
                for start, end in geo_blocks:
                    block_mask = torch.rand(batch_size, 1, device=device) > self.geo_block_dropout
                    geo_mask[:, start:end] = geo_mask[:, start:end] * block_mask.float()
            
            geo_condition = geo_condition * geo_mask

        # 2. 属性条件：双层缺失（整体缺失 + 单维度独立缺失）
        if self.use_tensor_condition and tensor_feature is not None:
            attr_dim = tensor_feature.shape[1]
            # 第一层：整体缺失掩码（整个属性条件都没有）
            global_mask = torch.rand(batch_size, 1, device=device) > self.tensor_global_dropout
            # 第二层：单维度缺失掩码（整体保留的前提下，每个属性独立随机缺失）
            per_dim_mask = torch.rand(batch_size, attr_dim, device=device) > self.tensor_per_dim_dropout
            # 合并掩码
            final_tensor_mask = global_mask.float() * per_dim_mask.float()
            tensor_feature = tensor_feature * final_tensor_mask
        # =================================

        loss = self.model.training_loss(
            occupancy,
            geo_condition,
            tensor_feature,
            self.global_step,
            self.training_epoch * self.iterations
        ).mean()

        self.log(
            "loss", loss.clone().detach().item(),
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True,
            batch_size=self.batch_size
        )

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        opt.step()

        self.update_EMA()
        return loss

    def validation_step(self, batch, batch_idx):
        occupancy = batch["occupancy"]
        geo_condition = batch["condition"]
        tensor_feature = batch["tensor_feature"] if self.use_tensor_condition else None

        val_loss = self.model.training_loss(
            occupancy,
            geo_condition,
            tensor_feature,
            self.training_epoch * self.iterations,
            self.training_epoch * self.iterations
        ).mean()

        self.log(
            "val_loss", val_loss.item(),
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True,
            batch_size=self.val_batch_size,
            sync_dist=True
        )
        return val_loss

    def on_train_epoch_end(self):
        self.log("current_epoch", self.current_epoch, logger=True)
        return super().on_train_epoch_end()