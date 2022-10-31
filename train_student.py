import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from dataset import boe
from dataset import oct2
from criterion.criterion import DistillKL
from helper.utils import adjust_learning_rate,  model_name_parser
from helper.loops import train_ST_KD as train, validate_ST_KD as validate


def student_parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int,
                        default=50, help='print frequency')
    parser.add_argument('--tb_freq', type=int,
                        default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int,
                        default=400, help='save frequency')
    parser.add_argument('--batch_size', type=int,
                        default=50, help='batch_size')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int,
                        default=80, help='number of training epochs')
    parser.add_argument('--info', type=str, default='', help='more infomation')

    # optimization
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='25,60', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model_s', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'resnext50_32x4d', 'resnext101_32x8d',
                                 'wide_resnet50_2', 'wide_resnet101_2'])

    parser.add_argument('--path_t', type=str, default="./save/models/FT_S_resnet50_T_resnet50_oct2_/resnet50_best.pth",
                        help='teacher model checkpoint')
    parser.add_argument('--d_rep', type=int, default=128,
                        help="dimension of representation layer")
    parser.add_argument('--dataset', type=str, default='oct2',
                        choices=['oct2', "boe"], help='dataset')
    # parser.add_argument('-T', '--temperature', type=float,
    #                     default=10, help='temperature')
    parser.add_argument('-a', '--alpha', type=float,
                        default=0.6, help='alpha multiplier')
    parser.add_argument('-b', '--beta', type=float,
                        default=0.4, help='weight for classification')

    parser.add_argument('-t', '--trial', type=int,
                        default=101, help='the experiment id')
    parser.add_argument('--parallel_training', type=bool, default=False)

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4,
                        help='temperature for KD distillation')

    parser.add_argument('--distill', type=str,
                        default='kd', choices=['kd', 'crd'])
    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('siweimai'):
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = model_name_parser(opt.path_t)

    opt.model_name = 'STKD{}_S_{}_T_{}_{}_a{}_b{}_KDT{}'.format(
        opt.trial, opt.model_s, opt.model_t, opt.dataset, opt.alpha, opt.beta, opt.kd_T)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if os.path.isdir(opt.tb_folder):
        opt.model_name = opt.model_name+"_"
        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        os.makedirs(opt.tb_folder)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if os.path.isdir(opt.save_folder):
        opt.model_name = opt.model_name+"_"
        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        os.makedirs(opt.save_folder)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = student_parse_option()

    print(opt.path_t)

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    n_classes = 0
    if opt.dataset == 'oct2':
        train_loader, val_loader = oct2.get_oct2_dataloaders_sub(
            batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_classes = 5
    elif opt.dataset == 'boe':
        train_loader, val_loader = boe.get_boe_dataloaders(
            batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_classes = 3
    else:
        raise NotImplementedError(opt.dataset)

    print("train set length:{}".format(len(train_loader.dataset)))
    print("test set length:{}".format(len(val_loader.dataset)))

    # * Teacher model part
    print('==> loading teacher model')
    base_name = model_name_parser(opt.path_t)
    base = model_dict["resnet50"](num_classes=opt.d_rep, input_channel=1)
    model_t = model_dict['rep_net'](
        base_net=base, d_rep=opt.d_rep, n_classes=n_classes)
    model_t.load_state_dict(torch.load(opt.path_t)['model'])
    print("==> {} based rep model loaded!".format(base_name))
    # ? >>>>>>>>>>> change the classification layer <<<<<<<<<<<
    model_t.linear = torch.nn.Linear(
        in_features=opt.d_rep, out_features=n_classes)

    # * Student model part
    model_s = model_dict[opt.model_s](num_classes=n_classes, input_channel=1)

    data = torch.randn(2, 1, 224, 224)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t.base_net(data, need_feat=True)
    feat_s, _ = model_s(data, need_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_div)

    # optimizer
    optimizer = optim.Adam(trainable_list.parameters(),
                           lr=opt.learning_rate,
                           weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    teacher_acc, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("-"*25)
        print("==> training...")

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(now_time)

        time1 = time.time()
        train_acc, train_loss = train(
            epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, (time2 - time1)/60))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_loss = validate(
            val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(
                opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model with new acc {}'.format(best_acc))
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(
        opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
