import argparse
import os
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from criterion.criterion import CrossEntropy_SNNL
from dataset import kaggle
from helper.loops import train_SNNL as train
from helper.loops import validate_SNNL as validate
from helper.utils import adjust_learning_rate
from models import model_dict


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=100, help="print frequency")
    parser.add_argument("--tb_freq", type=int, default=500, help="tb frequency")
    parser.add_argument("--save_freq", type=int, default=400, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=45, help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="number of training epochs"
    )

    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="10,20",
        help="where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument(
        "--model",
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
        "--d_rep", type=int, default=128, help="dimension of representation layer"
    )
    parser.add_argument(
        "--dataset", type=str, default="kaggle", choices=["kaggle"], help="dataset"
    )
    parser.add_argument(
        "-T", "--temperature", type=float, default=50, help="temperature"
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=-5.0, help="alpha multiplier"
    )
    parser.add_argument("-c", "--check-model", default=False,  action="store_true")
    parser.add_argument("-t", "--trial", type=int, default=0, help="the experiment id")
    parser.add_argument("--parallel-training", type=bool, default=False)

    parser.add_argument("--info", type=str, default="", help="more infomation")

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model in ["MobileNetV2", "ShuffleV1", "ShuffleV2"]:
        opt.learning_rate = 0.01

    opt.model_path = "./save/models"
    opt.tb_path = "./save/tensorboard"

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = [int(it) for it in iterations]
    opt.model_name = f"Teacher_{opt.model}_epochs{opt.epochs}_alpha{opt.alpha}_T{opt.temperature}_{opt.info}"

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():  # sourcery skip: use-fstring-for-formatting

    best_acc = 0

    opt = parse_option()
    print("a={},T={}".format(opt.alpha, opt.temperature))

    # dataset check
    if opt.dataset != "kaggle":
        raise NotImplementedError(opt.dataset)
    train_loader, val_loader = kaggle.get_kaggle_dataloaders(
        batch_size=opt.batch_size, num_workers=opt.num_workers
    )

    n_classes = 4
    # model
    base = model_dict[opt.model](input_channel=1, num_classes=opt.d_rep)
    model = model_dict["rep_net"](base, opt.d_rep, n_classes)
    # *check model
    if opt.check_model:
        summary(model, input_size=(32, 1, 224, 224), device=torch.device("cuda:0"))
        exit(0)
    # exit()
    # optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay
    )

    criterion = CrossEntropy_SNNL(T=opt.temperature, alpha=opt.alpha)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    assert torch.cuda.is_available(), "Not with GPU"

    base = base.to(device)
    model = model.to(device)
    criterion = criterion.to(device)
    torch.backends.cudnn.benchmark = True

    if opt.parallel_training:
        model = nn.DataParallel(model)

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(now_time)

        time1 = time.time()
        train_acc, train_loss = train(
            epoch, train_loader, model, criterion, optimizer, device, opt
        )
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, (time2 - time1) / 60))

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
            save_file = os.path.join(opt.save_folder, "{}_best.pth".format(opt.model))
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

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print("best accuracy:", best_acc)

    # save model
    state = {
        "opt": opt,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, "{}_last.pth".format(opt.model))
    torch.save(state, save_file)


if __name__ == "__main__":
    main()
