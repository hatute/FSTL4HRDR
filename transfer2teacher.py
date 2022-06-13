import argparse
import os
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from criterion.criterion import CrossEntropy_SNNL
from dataset import boe
from dataset import oct2
from helper.loops import train_SNNL as train, validate_SNNL as validate
from helper.utils import (
    adjust_learning_rate,
    load_model,
    freeze,
    model_name_parser,
    part_freeze,
)
from models import model_dict


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb_freq", type=int, default=10, help="tb frequency")
    parser.add_argument("--save_freq", type=int, default=100, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=45, help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="number of training epochs"
    )
    parser.add_argument("--info", type=str, default="", help="more infomation")

    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="20,40",
        help="where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument(
        "--model_s",
        type=str,
        default="resnet50",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "wrn_16_1",
            "wrn_16_2",
            "wrn_40_1",
            "wrn_40_2",
            "vgg8",
            "vgg11",
            "vgg13",
            "vgg16",
            "vgg19",
            "MobileNetV2",
            "ShuffleV1",
            "ShuffleV2",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2",
        ],
    )

    parser.add_argument(
        "--path_t",
        type=str,
        default="./save/models/Teacher_resnet50_epochs30_alpha-5.0_T50_a=-5/resnet50_best.pth",
        help="teacher model checkpoint",
    )
    parser.add_argument(
        "--d_rep", type=int, default=128, help="dimension of representation layer"
    )
    parser.add_argument(
        "--dataset", type=str, default="oct2", choices=["oct2", "boe"], help="dataset"
    )
    parser.add_argument(
        "-T", "--temperature", type=float, default=50, help="temperature"
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=-5, help="alpha multiplier"
    )
    parser.add_argument("-t", "--trial", type=int, default=0, help="the experiment id")
    parser.add_argument("--parallel_training", type=bool, default=False)

    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["ALL", "FE", "HL"],
        help="transfer learning method for teacher model",
    )
    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ["MobileNetV2", "ShuffleV1", "ShuffleV2"]:
        opt.learning_rate = 0.01

    # set the path according to the environment

    opt.model_path = "./save/models"
    opt.tb_path = "./save/tensorboard"

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = [int(it) for it in iterations]
    opt.model_t = model_name_parser(opt.path_t)

    method = opt.method

    if method == "ALL":
        opt.model_name = f"ALL_S_{opt.model_s}_T_{opt.model_t}_{opt.dataset}_{opt.info}"

    elif method == "FE":
        opt.model_name = f"FE_S_{opt.model_s}_T_{opt.model_t}_{opt.dataset}_{opt.info}"

    elif method == "HL":
        opt.model_name = f"HL_S_{opt.model_s}_T_{opt.model_t}_{opt.dataset}_{opt.info}"
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = parse_option()

    print(opt.path_t)

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == "oct2":
        train_loader, val_loader = oct2.get_oct2_dataloaders(
            batch_size=opt.batch_size, num_workers=opt.num_workers
        )
        n_classes = 5

    elif opt.dataset == "boe":
        train_loader, val_loader = boe.get_boe_dataloaders(
            batch_size=opt.batch_size, num_workers=opt.num_workers
        )
        n_classes = 3
    else:
        raise NotImplementedError(opt.dataset)

    print(f"train set length:{len(train_loader.dataset)}")
    print(f"test set length:{len(val_loader.dataset)}")
    # exit()
    # Teacher model
    # model_t = load_teacher(opt.path_t, n_classes)
    method = opt.method

    if method == "ALL":
        model = load_model(
            model_dict=model_dict, model_path=opt.path_t, d_rep=opt.d_rep, n_cls=4
        )

        model.linear = torch.nn.Linear(in_features=opt.d_rep, out_features=n_classes)
    elif method == "HL":
        model = load_model(
            model_dict=model_dict, model_path=opt.path_t, d_rep=opt.d_rep, n_cls=4
        )
        freeze(model)
        model.linear = torch.nn.Linear(in_features=opt.d_rep, out_features=n_classes)
    elif method == "FE":
        model = load_model(
            model_dict=model_dict, model_path=opt.path_t, d_rep=opt.d_rep, n_cls=4
        )

        model.linear = torch.nn.Linear(in_features=opt.d_rep, out_features=n_classes)

        # check_parameters_to_train(model)
        # for (i, (n, p)) in enumerate(model.named_parameters()):
        #     print(i, n)
        # 129
        part_freeze(model, 129)

    # check_parameters_to_train(model)

    optimizer = optim.Adam(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay
    )

    criterion = CrossEntropy_SNNL(T=opt.temperature, alpha=opt.alpha)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    assert torch.cuda.is_available(), "Not with GPU"

    model = model.to(device)
    criterion = criterion.to(device)
    cudnn.benchmark = True

    if opt.parallel_training:
        model = nn.DataParallel(model)

    # routine
    print("==> Start training...")
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        time1 = time.time()
        train_acc, train_loss = train(
            epoch, train_loader, model, criterion, optimizer, device, opt,
        )
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        logger.log_value("train_acc", train_acc, epoch)
        logger.log_value("train_loss", train_loss, epoch)

        # test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
        test_acc, test_loss = validate(val_loader, model, criterion, opt)
        logger.log_value("test_acc", test_acc, epoch)
        # logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value("test_losss", test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, f"{opt.model_s}_best.pth")
            print("saving the best model!")
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print("==> Saving...")
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "accuracy": test_acc,
                "optimizer": optimizer.state_dict(),
            }
            save_file = os.path.join(
                opt.save_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
        print("-" * 30)
    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print("best accuracy:", best_acc)
    end_time = time.time()
    print(time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time)))
    # save model
    state = {
        "opt": opt,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, f"{opt.model_s}_last.pth")
    torch.save(state, save_file)


if __name__ == "__main__":
    main()
