import timm
import torch
import torch.nn as nn
import numpy as np
import random

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from finetuning.main_worker import main_worker
from finetuning.dataloader import build_dataloader
from finetuning.models import create_vit_for_data2vec
from finetuning.scheduling import cosine_scheduler

class dotdict(dict):
    """
    Dictionary with dot access to it's fields.
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main(model_name):
    config = dotdict({
        "seed": 1337,
        "checkpoint_path": "checkpoints/checkpoint-437500/pytorch_model.bin",
        "data_path": "/mnt/data/imagenet/",
        "batch_size": 512,
        "req_batch_size": 1024,
        "n_epochs": 100,
        "device_ids": [0, 1],
        "warmup_epochs": 10,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "criterion": "soft_target_cross_entropy",
        "lr": 4e-3,
        "layer_decay": 0.65,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "lr_min": 1e-5,
        "warmup_lr_init": 1e-6,
        "use_scaler": False,
        "n_classes": 1000,
        # Augmentations
        "mixup_alpha": 0.8,
        "cutmix_alpha": 1.0,
        "cutmix_minmax": None,
        "mixup_prob": 1.0,
        "mixup_switch_prob": 0.5,
        "mixup_mode": "batch",
        "label_smoothing": 0.1,
    })

    print(config)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    model = create_vit_for_data2vec(config.checkpoint_path)

    req_num_iterations = int(config.req_batch_size / config.batch_size)

    device = torch.device(f"cuda:{config.device_ids[0]}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model.cuda(device)
    model = nn.DataParallel(model, device_ids=config.device_ids, output_device=device)

    train_loader, val_loader = build_dataloader(config.batch_size, config.data_path)

# Layerwise learning rate
    if getattr(config, "layer_decay", False):
        n_groups = model.module.get_num_param_groups()
        scales = [config.layer_decay ** (n_groups + 1 - i) for i in range(n_groups)]

        parameter_groups = [{'params': p, 'lr_scale': scale} 
                            for p, scale in zip(model.module.get_param_groups(), scales)]
    else:
        parameter_groups = model.parameters()

# Optimizer
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameter_groups, eps=config.eps, betas=config.betas, 
                                      lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise RuntimeError(f"{config.optimizer} is not supported yet. Please use 'adamw' optmizer.")

# Criterion
    if config.criterion == "soft_target_cross_entropy":
        criterion = SoftTargetCrossEntropy().cuda(device)
    else:
        raise RuntimeError(f"{config.criterion} is not supported yet. Please use 'soft_target_cross_entropy' criterion.")

# Scheduler
    if config.scheduler == "cosine":
        schedule = cosine_scheduler(
            config.lr,
            config.lr_min,
            config.n_epochs,
            len(train_loader) // req_num_iterations,
            config.warmup_epochs,
            config.warmup_lr_init,
        )
    else:
        raise RuntimeError(f"{config.scheduler} is not supported yet. Please use 'cosine' schedule.")

# Loss scaler
    if config.use_scaler:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
        
    mixup_fn = Mixup(mixup_alpha=config.mixup_alpha, cutmix_alpha=config.cutmix_alpha, cutmix_minmax=config.cutmix_minmax, 
                     prob=config.mixup_prob, switch_prob=config.mixup_switch_prob, mode=config.mixup_mode,
                     label_smoothing=config.label_smoothing, num_classes=config.n_classes)
    start_epoch = -1
    saved_model_path = None #"/home/meshchaninov/MLBEP/saved_models/swin-v-1.2_20220410/Epoch: 296, acc: 0.79867"
    
    if saved_model_path:
        state = torch.load(saved_model_path)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]

    main_worker(config.n_epochs, model, optimizer, criterion, schedule, scaler, model_name, train_loader, val_loader,
                req_num_iterations, device, mixup_fn, config, start_epoch)


if __name__ == '__main__':
    main("vit_data2vec")
