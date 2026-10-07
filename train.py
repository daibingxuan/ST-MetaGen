import fire
import os

from network.model_trainer import DiffusionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from utils.utils import exists
from pytorch_lightning import loggers as pl_loggers
from utils.utils import ensure_directory, run, get_tensorboard_dir, find_best_epoch
from torch.utils.tensorboard import SummaryWriter
import torch
import pdb

torch.set_num_threads(2)

def train_from_folder(
    dataset_folder: str = "/home/daibingxuan/workspace/metaset/3D_diverse/2020_JMD/shape_generation_code/metaset_64_npy",
    results_folder: str = './results',
    voxel_folder: str = "/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/randombulk_compress",
    csv_path: str = "/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/voxel_descriptions_npr.csv",
    name: str = "debug",
    image_size: int = 64,
    base_channels: int = 64,
    optimizier: str = "adam",
    attention_resolutions: str = "4, 8",
    lr: float = 2e-4,
    batch_size: int = 4,
    with_attention: bool = True,
    num_heads: int = 4,
    dropout: float = 0.1,
    noise_schedule: str = "linear",
    ema_rate: float = 0.999,
    verbose: bool = False,
    training_epoch: int = 200,
    in_azure: bool = False,
    new: bool = True,
    continue_training: bool = False,
    debug: bool = False,
    # ========== 几何条件与属性条件配置 ==========
    use_geo_condition: bool = True,
    cond_type: str = "onehot",
    geo_condition_dim: int = 50,
    use_tensor_condition: bool = False,
    tensor_condition_dim: int = 3,
    # ========== 条件缺失配置（新增） ==========
    geo_cond_dropout: float = 0.1,          # 几何条件整体缺失概率
    geo_block_dropout: float = 0.15,        # 几何条件语义分块独立缺失概率
    tensor_global_dropout: float = 0.2,     # 属性条件整体缺失概率
    tensor_per_dim_dropout: float = 0.25,   # 属性条件单维度缺失概率
    # ========== 验证集配置 ==========
    use_validation: bool = True,
    val_batch_size: int = 8,
    val_ratio: float = 0.2,
    # =================================
    seed: int = 777,
    gradient_clip_val: float = 1.,
    # 手动指定续训权重文件，为空则自动查找
    resume_ckpt: str = "",
):
    if not in_azure:
        debug = True
    else:
        debug = False

    # 创建结果文件夹
    results_folder = results_folder + "/" + name
    ensure_directory(results_folder)
    if continue_training:
        new = False
    if new:
        run(f"rm -rf {results_folder}/*")

    # 超参数字典（与 DiffusionModel 参数一一对应）
    model_args = dict(
        results_folder=results_folder,
        dataset_folder=dataset_folder,
        voxel_folder=voxel_folder,
        csv_path=csv_path,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        noise_schedule=noise_schedule,
        # 几何条件配置
        use_geo_condition=use_geo_condition,
        cond_type=cond_type,
        geo_condition_dim=geo_condition_dim,
        # 弹性属性配置
        use_tensor_condition=use_tensor_condition,
        tensor_condition_dim=tensor_condition_dim,
        # 条件缺失配置（新增）
        geo_cond_dropout=geo_cond_dropout,
        geo_block_dropout=geo_block_dropout,
        tensor_global_dropout=tensor_global_dropout,
        tensor_per_dim_dropout=tensor_per_dim_dropout,
        # 验证集配置
        use_validation=use_validation,
        val_batch_size=val_batch_size,
        val_ratio=val_ratio,
        # 网络结构
        base_channels=base_channels,
        optimizier=optimizier,
        attention_resolutions=attention_resolutions,
        with_attention=with_attention,
        num_heads=num_heads,
        dropout=dropout,
        ema_rate=ema_rate,
        verbose=verbose,
        training_epoch=training_epoch,
        gradient_clip_val=gradient_clip_val,
        debug=debug,
    )
    seed_everything(seed)

    model = DiffusionModel(**model_args)

    # 日志目录
    if in_azure:
        try:
            log_dir = get_tensorboard_dir()
        except Exception as e:
            log_dir = results_folder
    else:
        log_dir = results_folder

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version=None,
        name='logs',
        default_hp_metric=False
    )

    # 仅保留最优模型 checkpoint
    monitor_metric = "val_loss" if use_validation else "loss"
    best_loss_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=results_folder,
        filename=f"best-{monitor_metric}-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        save_top_k=1,
        save_last=True,
        mode="min"
    )

    # 自动查找续训权重
    auto_last_ckpt = None
    if os.path.isdir(results_folder):
        for f in os.listdir(results_folder):
            if f.startswith(f"best-{monitor_metric}") and f.endswith(".ckpt"):
                auto_last_ckpt = f
                break

    # 优先级：手动指定 > 自动查找
    if resume_ckpt:
        last_ckpt = resume_ckpt
    else:
        last_ckpt = auto_last_ckpt

    # 备份权重
    last_ckpt_path = os.path.join(results_folder, last_ckpt) if last_ckpt else ""
    if last_ckpt is not None:
        backup_ckpt_path = os.path.join(results_folder, "best_backup.ckpt")
        if continue_training and os.path.exists(last_ckpt_path) and not os.path.exists(backup_ckpt_path):
            import shutil
            shutil.copyfile(last_ckpt_path, backup_ckpt_path)
            print(f"Backup of best checkpoint saved to {backup_ckpt_path}")

    # 初始化训练器
    find_unused_parameters = False
    trainer_kwargs = dict(
        devices=-1,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=find_unused_parameters),
        logger=tb_logger,
        max_epochs=training_epoch,
        callbacks=[best_loss_checkpoint]
    )
    if in_azure:
        trainer_kwargs["log_every_n_steps"] = 10
    else:
        trainer_kwargs["log_every_n_steps"] = 1

    trainer = Trainer(**trainer_kwargs)

    # 启动训练
    if continue_training and last_ckpt is not None and os.path.exists(last_ckpt_path):
        trainer.fit(model, ckpt_path=last_ckpt_path)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(train_from_folder)