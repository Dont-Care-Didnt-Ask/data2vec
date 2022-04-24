import time

import torch
from tqdm import tqdm
import wandb

from finetuning.metrics import accuracy


def train(train_loader, model, criterion, optimizer, schedule, epoch, device, writer_train,
          req_num_iterations, mixup_fn):
    print_freq = 100

    # switch to train mode
    model.train()
    optimizer.zero_grad()
    loss_epoch = 0.0
    acc_epoch = 0.0
    count_els = 0

    T = tqdm(enumerate(train_loader), desc=f'epoch {epoch}', position=0, leave=True)

    end = time.time()
    for i, (images, target) in T:
        # measure data loading time
        data_time = time.time() - end
        images = images.cuda(device)
        target = target.cuda(device)
        images, target = mixup_fn(images, target)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        (loss / req_num_iterations).backward()

        # measure accuracy
        acc = accuracy(output, torch.argmax(target, dim=1))

        # compute gradient and do SGD step
        if (i + 1) % req_num_iterations == 0:
            #scheduler.step_update(epoch * len(train_loader) + i) -- substituted with manual scheduling

            if schedule is not None:
                it = (len(train_loader) // req_num_iterations) * epoch + i // req_num_iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = schedule[it] * param_group['lr_scale']

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()

        loss_epoch += loss.item() * target.size(0)
        acc_epoch += acc.item() * target.size(0)
        count_els += target.size(0)

        T.set_description(f"Epoch {epoch}, loss: {loss_epoch / count_els:.5f}, " + \
                          f"accuracy: {acc_epoch / count_els:.5f}, data_time: {data_time:.3f}, " + \
                          f"batch_time: {batch_time:.3f}, lr_embed: {optimizer.param_groups[0]['lr']:.7f}, " + \
                          f"lr_head: {optimizer.param_groups[-1]['lr']:.7f}",
                          refresh=False)

        if (i + 1) % print_freq == 0:
            training_iter = epoch * (len(train_loader) // req_num_iterations) + i // req_num_iterations
            writer_train.add_scalar('train_loss', loss_epoch / count_els, global_step=training_iter)
            writer_train.add_scalar('train_accuracy', acc_epoch / count_els, global_step=training_iter)

            wandb.log({
                'training_iter': training_iter,
                'train_loss': loss_epoch / count_els,
                'train_accuracy': acc_epoch / count_els,
                'learning_rate_zero_group': optimizer.param_groups[0]['lr'],
                'learning_rate_last_group': optimizer.param_groups[-1]['lr']
            })