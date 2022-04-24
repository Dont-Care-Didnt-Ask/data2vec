import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import wandb

from finetuning.train import train
from finetuning.validation import validate


def main_worker(num_epochs, model, opt, criterion, schedule, scaler, model_name, train_loader, val_loader,
                req_batch_size, device, mixup_fn, config, start_epoch=0):
    log_dir = f"/home/myyycroft/repos/data2vec/outputs/{model_name}_{datetime.datetime.now().strftime('%Y_%m_%d_%Hh')}"
    save_dir = f"/home/myyycroft/repos/data2vec/checkpoints/{model_name}_{datetime.datetime.now().strftime('%Y_%m_%d_%Hh')}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(os.path.join(log_dir, 'valid'))

    experiment_name = f"{model_name}_finetune_{datetime.datetime.now().strftime('%Y_%m_%d_%Hh')}"
    wandb.init(project="data2vec", entity="exxxplainer", name=experiment_name, config=config)
    wandb.watch(model)

    best_acc1 = -1.
    for epoch in range(start_epoch + 1, num_epochs):
        train(train_loader, model, criterion, opt, schedule, epoch, device, writer_train, req_batch_size, mixup_fn)
        acc1 = validate(val_loader, model, epoch, device, writer_val)

        wandb.log({"validation_accuracy": acc1, "epoch": epoch})

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': opt.state_dict(),
                'schedule': schedule,
            }
            if scaler is not None:
                state["scaler"] = scaler.state_dict()

            torch.save(state, f"{save_dir}/Epoch: {epoch}, acc: {best_acc1:.5f}")
