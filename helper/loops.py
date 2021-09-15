import sys
import time
import torch
import pandas as pd
from .utils import AverageMeter, accuracy
import numpy as np


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        if opt.model_s == "inception_v3":
            output = model(input)
            output = output
        else:
            output = model(input)
        loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1, ))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg


def validate_vanilla(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                          idx, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg,  losses.avg


def validate_student(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    df = pd.DataFrame(columns=['target', 'output', 'path'])
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target, path) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            # print(idx, path)
            # exit()

            # compute output
            output = model(input)
            tgt = target.cpu().numpy()
            otpt = torch.argmax(output, dim=1).cpu().numpy()
            rst = np.array([tgt, otpt, path]).transpose(1, 0)
            df = df.append(pd.DataFrame(
                rst, columns=['target', 'output', 'path']), ignore_index=True)
            # print(df)
            # print()
            # exit()

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                          idx, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
    print(len(df))
    return df


def validate_ST_KD(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            # ! rep_net have two output: y_ and base_net

            output = model(input)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                          idx, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg,  losses.avg


def train_ST_KD(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()

    # print(module_list)
    # exit()

    # set teacher as eval()
    module_list[-1].eval()

    # if opt.distill == 'abound':
    #     module_list[1].eval()
    # elif opt.distill == 'factor':
    #     module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):

        input, target = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        logit_s = model_s(input, need_feat=False)

        with torch.no_grad():
            logit_t, _ = model_t(input)

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        loss = opt.alpha * loss_cls + opt.beta * loss_div

        acc1 = accuracy(logit_s, target, topk=(1, ))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f},{l_cls:.3f}+{l_div:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, l_cls=loss_cls, l_div=loss_div, top1=top1))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} '
          .format(top1=top1))

    return top1.avg, losses.avg


def train_SNNL(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top3 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output, x_rep = model(input)
        # print(model)
        # print([i.shape for i in model(input)])
        # exit()
        loss = criterion(x_rep, output, target)

        # measure accuracy and record loss
        # acc1, acc3 = accuracy(output, target, topk=(1, 3))
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        # top3.update(acc3[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass
        # print(idx)
        # print(opt.print_freq)
        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f}'
          .format(top1=top1))
    #
    return top1.avg, losses.avg


def validate_SNNL(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output, x_rep = model(input)
            loss = criterion(x_rep, output, target)

            # measure accuracy and record loss
            # acc1, acc3 = accuracy(output, target, topk=(1, 3))
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            # top3.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if idx % opt.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #           'Acc@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
            #         idx, len(val_loader), batch_time=batch_time, loss=losses,
            #         top1=top1, top3=top3))

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          idx, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg, losses.avg
