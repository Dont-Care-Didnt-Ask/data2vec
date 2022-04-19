import timm
import torch
import torch.nn as nn
import numpy as np
import random

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from main_worker import main_worker
from dataloader import build_dataloader


def main(model_name):
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = timm.create_model(model_name="swin_tiny_patch4_window7_224", pretrained=False)

    batch_size = 1024
    req_batch_size = 1024
    req_num_iterations = int(req_batch_size / batch_size)
    num_epochs = 300

    device_ids = [0, 1, 2, 3]
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model.cuda(device)
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device)

    train_loader, val_loader = build_dataloader(batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=1e-3, weight_decay=0.05)
    criterion = SoftTargetCrossEntropy().cuda(device)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=int(num_epochs * len(train_loader)),
        lr_min=1e-5,
        warmup_lr_init=1e-6,
        warmup_t=int(20 * len(train_loader)),
        cycle_limit=1,
        t_in_epochs=False,
    )
    scaler = torch.cuda.amp.GradScaler()
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=1.0, switch_prob=0.5, mode='batch',
                     label_smoothing=0.1, num_classes=1000)
    start_epoch = -1
    saved_model_path = None #"/home/meshchaninov/MLBEP/saved_models/swin-v-1.2_20220410/Epoch: 296, acc: 0.79867"
    if saved_model_path:
        state = torch.load(saved_model_path)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]

    main_worker(num_epochs, model, optimizer, criterion, scheduler, scaler, model_name, train_loader, val_loader,
                req_num_iterations, device, mixup_fn, start_epoch)


if __name__ == '__main__':
    main("swin-v-1.3-timm")
