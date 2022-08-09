from torchmetrics import ConfusionMatrix
import os
from models import model_dict
import argparse
from dataset import oct2
from helper.loops import validate_return_raw_4CM as validate
import torch

from sklearn.metrics import confusion_matrix


def cm_parse_option():

    parser = argparse.ArgumentParser("test for confusion matrix")

    parser.add_argument("--n_classes", type=int, default=5)
    # datase
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
        "--dataset", type=str, default="zs", choices=["zs", "hb"], help="dataset"
    )
    parser.add_argument(
        "--path_s",
        type=str,
        default="/data/local/siwei/workspace/FSTL4HRDR/save/models/STKD101_S_resnet18_T_resnet50_zszs_a0.6_b0.4_KDT4_(Pzs_2)/F5_resnet18_best.pth",
    )
    return parser.parse_args()


def main():
    opt = cm_parse_option()
    train_loader, val_loader = oct2.get_oct2_dataloaders(c_dataset=opt.dataset)
    # print(len(val_loader))
    model_s = model_dict[opt.model_s](num_classes=opt.n_classes, input_channel=1).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_s.load_state_dict(torch.load(opt.path_s)["model"])

    target, output = validate(val_loader=val_loader, model=model_s)
    print(f"tgt_len:{len(target)}, output_len:{len(output)}")
    print(confusion_matrix(output, target))
    print("HW")


if __name__ == "__main__":
    main()
