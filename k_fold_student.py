import os

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST
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
from helper.utils import adjust_learning_rate, model_name_parser
from helper.loops import train_ST_KD as train, validate_ST_KD as validate


def student_parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--tb_freq", type=int, default=500, help="tb frequency")
    parser.add_argument("--save_freq", type=int, default=500, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="number of training epochs"
    )
    parser.add_argument("--info", type=str, default="", help="more infomation")

    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="25,60",
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
        default="resnet18",
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
        default="./save/models/FT_S_resnet50_T_resnet50_oct2_/resnet50_best.pth",
        help="teacher model checkpoint",
    )
    parser.add_argument(
        "--d_rep", type=int, default=128, help="dimension of representation layer"
    )
    # parser.add_argument('--dataset', type=str, default='oct2',
    #                     choices=['oct2', "boe"], help='dataset')
    # parser.add_argument('-T', '--temperature', type=float,
    #                     default=10, help='temperature')
    parser.add_argument(
        "-a", "--alpha", type=float, default=0.5, help="alpha multiplier"
    )
    parser.add_argument(
        "-b", "--beta", type=float, default=0.5, help="weight for classification"
    )

    parser.add_argument(
        "-t", "--trial", type=int, default=101, help="the experiment id"
    )
    parser.add_argument("--parallel_training", type=bool, default=False)

    # KL distillation
    parser.add_argument(
        "--kd_T", type=float, default=4, help="temperature for KD distillation"
    )

    parser.add_argument("--distill", type=str, default="kd", choices=["kd", "crd"])
    # dataset choice
    parser.add_argument("--train_dataset", type=str, choices=["hb", "zs"])
    parser.add_argument("--test_dataset", type=str, choices=["hb", "zs"])
    parser.add_argument("-K", "--k_fold", type=int, default=5, help="number of k fold")

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ["MobileNetV2", "ShuffleV1", "ShuffleV2"]:
        opt.learning_rate = 0.01

    # set the path according to the environment

    opt.model_path = "./save/models"
    opt.tb_path = "./save/tensorboard"

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = model_name_parser(opt.path_t)

    opt.model_name = f"STKD{opt.trial}_S_{opt.model_s}_T_{opt.model_t}_{opt.train_dataset}{opt.test_dataset}_a{opt.alpha}_b{opt.beta}_KDT{opt.kd_T}_{opt.info}"

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if os.path.isdir(opt.tb_folder):
        opt.model_name = opt.model_name + "_"
        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        os.makedirs(opt.tb_folder)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if os.path.isdir(opt.save_folder):
        opt.model_name = opt.model_name + "_"
        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        os.makedirs(opt.save_folder)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


def main():
    torch.manual_seed(42)

    opt = student_parse_option()
    print(f"Teacher model path:{opt.path_t}")
    n_classes = 5
    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    print("==> loading teacher model")
    base_name = model_name_parser(opt.path_t)
    base = model_dict["resnet50"](num_classes=opt.d_rep, input_channel=1)
    model_t = model_dict["rep_net"](base_net=base, d_rep=opt.d_rep, n_classes=n_classes)
    model_t.load_state_dict(torch.load(opt.path_t)["model"])
    print("==> {} based rep model loaded!".format(base_name))
    # ? >>>>>>>>>>> change the classification layer <<<<<<<<<<<
    model_t.linear = torch.nn.Linear(in_features=opt.d_rep, out_features=n_classes)

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

    class_weight = oct2.get_class_weight(opt.train_dataset)
    print(f">>> class weight:{class_weight}")
    criterion_cls = nn.CrossEntropyLoss(weight=class_weight)
    criterion_div = DistillKL(opt.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_div)

    # optimizer
    optimizer = optim.Adam(
        trainable_list.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay
    )

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    assert torch.cuda.is_available(), "Not with GPU"

    module_list = module_list.to(device)
    criterion_list = criterion_list.to(device)
    cudnn.benchmark = True

    if opt.parallel_training:
        model = nn.DataParallel(module_list)

    if opt.train_dataset == opt.test_dataset:
        print("k-fold mode")
        k_fold(
            opt=opt,
            module_list=module_list,
            criterion_list=criterion_list,
            optimizer=optimizer,
            logger=logger,
        )
    else:
        print("hybird mode")
        hybird(
            opt=opt,
            module_list=module_list,
            criterion_list=criterion_list,
            optimizer=optimizer,
            logger=logger,
        )


def val_teacher(val_loader, model_t, criterion_cls, opt):
    teacher_acc, _ = validate(val_loader, model_t, criterion_cls, opt)
    print("teacher accuracy: ", teacher_acc)


def k_fold(opt, module_list, criterion_list, optimizer, logger):
    k = opt.k_fold
    results = {}
    criterion_cls = criterion_list[0]
    # add timer
    print("==> Start training...")
    start_time = time.time()

    for idx, (train_loader, val_loader) in enumerate(
        oct2.get_kfold_dataloader(
            k=k,
            c_dataset=opt.train_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
        )
    ):

        val_teacher(
            val_loader=val_loader,
            model_t=module_list[-1],
            criterion_cls=criterion_cls,
            opt=opt,
        )

        print(f"FOLD {idx + 1}/{k}")
        print("--------------------------------")
        # Init the neural network
        module_list[0].apply(reset_weights)
        best_acc = 0
        for epoch in range(1, opt.epochs + 1):

            adjust_learning_rate(epoch, opt, optimizer)

            now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(now_time)

            time1 = time.time()
            train_acc, train_loss = train(
                epoch, train_loader, module_list, criterion_list, optimizer, opt
            )
            time2 = time.time()
            print("epoch {}, total time {:.2f}".format(epoch, (time2 - time1) / 60))

            logger.log_value(f"F{idx + 1}_train_acc", train_acc, epoch)
            logger.log_value(f"F{idx + 1}_train_loss", train_loss, epoch)

            test_acc, test_loss = validate(
                val_loader, module_list[0], criterion_cls, opt
            )

            logger.log_value(f"F{idx + 1}_test_acc", test_acc, epoch)
            logger.log_value(f"F{idx + 1}_test_loss", test_loss, epoch)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    "epoch": epoch,
                    "model": module_list[0].state_dict(),
                    "best_acc": best_acc,
                }
                save_file = os.path.join(
                    opt.save_folder, f"F{idx + 1}_{opt.model_s}_best.pth"
                )
                print(f"saving the best model with new acc {best_acc}")
                torch.save(state, save_file)

            # regular saving
            if epoch % opt.save_freq == 0:
                print("==> Saving...")
                state = {
                    "epoch": epoch,
                    "model": module_list[0].state_dict(),
                    "accuracy": test_acc,
                }
                save_file = os.path.join(
                    opt.save_folder, f"F{idx + 1}_ckpt_epoch_{epoch}.pth"
                )
                torch.save(state, save_file)

        # This best accuracy is only for printing purpose.
        # The results reported in the paper/README is from the last epoch.
        print(f"F{idx + 1}_best accuracy:{best_acc}")
        results[idx + 1] = best_acc

        # save model
        state = {
            "opt": opt,
            "model": module_list[0].state_dict(),
        }
        save_file = os.path.join(opt.save_folder, f"F{idx + 1}{opt.model_s}_last.pth")
        torch.save(state, save_file)

    # Print final results
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {k} FOLDS")
    print("--------------------------------")
    acc_sum = 0.0
    for key, value in results.items():
        print(f"Fold {key}: {value} %")
        acc_sum += value
    print(f"Average: {acc_sum / len(results.items())} %")
    end_time = time.time()
    print(time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time)))


def hybird(opt, module_list, criterion_list, optimizer, logger):
    best_acc = 0
    criterion_cls = criterion_list[0]

    train_loader, val_loader = oct2.get_hybird_dataloaders(
        c_train=opt.train_dataset,
        c_test=opt.test_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
    )
    val_teacher(
        val_loader=val_loader,
        model_t=module_list[-1],
        criterion_cls=criterion_cls,
        opt=opt,
    )
    # add timer
    print("==> Start training...")
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("-" * 25)
        print("==> training...")

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(now_time)

        time1 = time.time()
        train_acc, train_loss = train(
            epoch, train_loader, module_list, criterion_list, optimizer, opt
        )
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, (time2 - time1) / 60))

        logger.log_value(
            f"TrainDS_{opt.train_dataset}_TestDS_{opt.test_dataset}_train_acc",
            train_acc,
            epoch,
        )
        logger.log_value(
            f"TrainDS_{opt.train_dataset}_TestDS_{opt.test_dataset}_train_loss",
            train_loss,
            epoch,
        )

        test_acc, test_loss = validate(val_loader, module_list[0], criterion_cls, opt)

        logger.log_value(
            f"TrainDS_{opt.train_dataset}_TestDS_{opt.test_dataset}_test_acc",
            test_acc,
            epoch,
        )
        logger.log_value(
            f"TrainDS_{opt.train_dataset}_TestDS_{opt.test_dataset}_test_loss",
            test_loss,
            epoch,
        )

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                "epoch": epoch,
                "model": module_list[0].state_dict(),
                "best_acc": best_acc,
            }
            save_file = os.path.join(
                opt.save_folder,
                f"TrainDS_{opt.train_dataset}_TestDS_{opt.test_dataset}_{opt.model_s}_best.pth",
            )
            print("saving the best model with new acc {}".format(best_acc))
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print("==> Saving...")
            state = {
                "epoch": epoch,
                "model": module_list[0].state_dict(),
                "accuracy": test_acc,
            }
            save_file = os.path.join(
                opt.save_folder,
                f"TrainDS_{opt.train_dataset}_TestDS_{opt.test_dataset}_{opt.model_s}_ckpt_epoch_{epoch}.pth",
            )
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print(
        f"TrainDS_{opt.train_dataset}_TestDS_{opt.test_dataset}_{opt.model_s}_best accuracy:",
        best_acc,
    )

    # save model
    state = {
        "opt": opt,
        "model": module_list[0].state_dict(),
    }
    save_file = os.path.join(
        opt.save_folder,
        f"TrainDS_{opt.train_dataset}_TestDS_{opt.test_dataset}_{opt.model_s}_last.pth",
    )
    torch.save(state, save_file)
    end_time = time.time()
    print(time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time)))


if __name__ == "__main__":
    main()
    # Configuration options
