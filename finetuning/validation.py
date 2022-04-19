import torch
from tqdm import tqdm
import time
import torch.nn as nn

from metrics import accuracy


def validate(val_loader, model, epoch, device, writer_val):
    criterion = nn.CrossEntropyLoss().cuda(device)
    loss_epoch = 0.0
    acc_epoch = 0.0
    count_els = 0
    T = tqdm(enumerate(val_loader), desc=f'epoch {epoch}', position=0, leave=True)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in T:
            data_time = time.time() - end
            images = images.cuda(device)
            target = target.cuda(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            loss_epoch += loss.item() * target.size(0)

            # measure accuracy and record loss
            acc = accuracy(output, target)
            acc_epoch += acc.item() * target.size(0)
            count_els += target.size(0)

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            T.set_description(f"Epoch: {epoch}, loss: {loss_epoch / count_els:.5f}, accuracy: {acc_epoch / count_els:.5f}, " + \
                              f"data_time: {data_time:.3f}, batch_time: {batch_time:.3f}", refresh=False)

    writer_val.add_scalar('test_loss', loss_epoch / count_els, global_step=epoch)
    writer_val.add_scalar('test_accuracy', acc_epoch / count_els, global_step=epoch)
    return acc_epoch / count_els
