import torch
import numpy as np
import os
import argparse
import socket


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


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


def model_name_parser(model_path):
    # print(model_path)
    model_name = model_path.split(os.sep)[-1].split("_")[0]
    print(model_name)
    return model_name


def load_model(model_dict, model_path, d_rep, n_cls):
    print('==> loading teacher model')
    base_name = model_name_parser(model_path)
    base = model_dict[base_name](num_classes=d_rep, input_channel=1)
    model = model_dict['rep_net'](base_net=base, d_rep=d_rep, n_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print("==> {} based rep model loaded!".format(base_name))
    return model


def check_parameters_to_train(freezed_model):
    print("Params to train:")
    params_to_update = []
    for name, param in freezed_model.named_parameters():
        if param.requires_grad:
            params_to_update.append(name)
    if len(params_to_update) > 0:
        for i in params_to_update:
            print("\t", i)
    else:
        print("Nothing to train")


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    check_parameters_to_train(model)


def part_freeze(model, start_index):
    for idx, (name, params) in enumerate(model.named_parameters()):
        if idx < start_index:
            params.requires_grad = False
