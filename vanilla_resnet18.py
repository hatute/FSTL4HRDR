import os
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
from models import model_dict

from dataset import oct2


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(epoch, train_loader, model, criterion, optimizer, device):
    """One epoch distillation"""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    # end = time.time()
    for idx, data in enumerate(train_loader):
        input, target = data
        input = input.float()
        input = input.to(device)
        target = target.to(device)
        # ===================forward=====================
        logit = model(input, need_feat=False)
        loss = criterion(logit, target)

        acc1 = accuracy(logit, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Train: [{epoch}]\n"
        f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
        f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
    )
    sys.stdout.flush()

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, device):
    """validation"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

    print(
        f"Test: \n"
        f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
        f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})\n -------------------- "
    )
    return top1.avg, losses.avg


def main():
    # TODO: CUDA availability check
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    assert torch.cuda.is_available(), "Not with GPU"

    timestamp = time.strftime("%y%m%d_%H%M%S")
    best_acc = 0
    epochs = 100
    model_path = "./save/models"
    tb_path = "./save/tensorboard"
    c_dataset = "oct4"
    n_classes = 5
    c_model = "resnet18"
    learning_rate = 1e-3
    model_arch = "rawOCT4"
    model_name = f"{model_arch}_{c_model}_{c_dataset}_{timestamp}"
    tb_folder = os.path.join(tb_path, model_name)
    save_folder = os.path.join(model_path, model_name)
    os.makedirs(tb_folder)
    os.makedirs(save_folder)
    logger = tb_logger.Logger(logdir=tb_folder, flush_secs=2)
    print(f"tensorboard will save to {tb_folder}")

    train_loader, val_loader = oct2.get_oct2_dataloaders_by_subject(c_dataset=c_dataset, raw=True, batch_size=64)

    model = model_dict[c_model](num_classes=n_classes, input_channel=1).to(device)
    criterion = nn.CrossEntropyLoss(weight=oct2.get_class_weight(c_dataset)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=15, min_lr=1e-6, verbose=True
    )

    # TODO: Training Process
    print("==> Start training...")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # ! train part
        time1 = time.time()
        train_acc, train_loss = train(
            epoch=epoch,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        logger.log_value(
            f"train_acc",
            train_acc,
            epoch,
        )
        logger.log_value(
            f"train_loss",
            train_loss,
            epoch,
        )

        test_acc, test_loss = validate(
            val_loader=val_loader, model=model, criterion=criterion, device=device
        )
        logger.log_value(
            f"test_acc",
            test_acc,
            epoch,
        )
        logger.log_value(
            f"test_loss",
            test_loss,
            epoch,
        )

        scheduler.step(test_loss)
        # ! save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "best_acc": best_acc,
            }
            save_file = os.path.join(
                save_folder,
                f"{model_arch}_best_{best_acc}.pth",
            )
            print("saving the best model with new acc {}".format(best_acc))
            torch.save(state, save_file)

        time2 = time.time()
        print(f"epoch {epoch}, total time {(time2 - time1) / 60:.2f}")

    print(f"{model_arch}'s best accuracy: {best_acc}")

    state = {
        "epoch": epochs,
        "model": model.state_dict(),
        "best_acc": best_acc,
    }
    save_file = os.path.join(
        save_folder,
        f"{model_arch}_last.pth",
    )
    torch.save(state, save_file)
    end_time = time.time()
    print(time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time)))


if __name__ == "__main__":

    main()
