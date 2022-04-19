import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

from train import train
from validation import validate


def main_worker(num_epochs, model, opt, criterion, scheduler, scaler, model_name, train_loader, val_loader,
                req_batch_size, device, mixup_fn, start_epoch=0):
    # Data loading code

    log_dir = f"/home/meshchaninov/MLBEP/tensorboard_logs/{model_name}_{datetime.datetime.now().strftime('%Y%m%d')}"
    save_dir = f"/home/meshchaninov/MLBEP/saved_models/{model_name}_{datetime.datetime.now().strftime('%Y%m%d')}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(os.path.join(log_dir, 'valid'))

    best_acc1 = -1
    for epoch in range(start_epoch + 1, num_epochs):
        train(train_loader, model, criterion, opt, scheduler, epoch, device, writer_train, req_batch_size, mixup_fn)
        acc1 = validate(val_loader, model, epoch, device, writer_val)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }

            torch.save(state, f"{save_dir}/Epoch: {epoch}, acc: {best_acc1:.5f}")
