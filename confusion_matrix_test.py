# from torchmetrics import ConfusionMatrix
import os
from models import model_dict
import argparse
from dataset import oct2
from helper.loops import validate_return_raw_4CM as validate
import torch

from sklearn.metrics import confusion_matrix
from torchmetrics import Specificity
from torchmetrics.functional import recall
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import auc


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
        "--dataset",
        type=str,
        default="oct4",
        choices=["zs", "hb", "oct4"],
        help="dataset",
    )
    parser.add_argument(
        "--path_s",
        type=str,
        # default="/data/local/siwei/workspace/FSTL4HRDR/save/models/STKD101_S_resnet18_T_resnet50_zszs_a0.6_b0.4_KDT4_(Pzs_2)/F5_resnet18_best.pth",
        # default="/data/local/siwei/workspace/FSTL4HRDR/save/models/VNL_ResNet18_zs_220817-223520/VNL_ResNet18_best.pth"
        # default="/data/local/siwei/workspace/FSTL4HRDR/save/models/VNL_ResNet18_hb_220817-223036/VNL_ResNet18_best.pth",
        # STL
        default="/data/local/siwei/workspace/FSTL4HRDR/save/models/STL_OCT4_resnet18_resnet50_T5.0_a0.15_oct4_221030_135345/STL_OCT4_best_85.4237289428711.pth",
        # raw_resnet18
        # default="/data/local/siwei/workspace/FSTL4HRDR/save/models/rawOCT4_ResNet18_oct4_221027-215841/rawOCT4_ResNet18_last.pth",
    )
    return parser.parse_args()


def main():
    opt = cm_parse_option()
    file_name = f"path_idx_STL_FF_{opt.dataset}.xlsx"
    STL = True
    # STL = False
    raw = True
    need_idx = True
    if opt.dataset == "oct4":
        train_loader, val_loader = oct2.get_oct2_dataloaders_by_subject(
            c_dataset=opt.dataset, raw=True, batch_size=64, need_idx=need_idx
        )
    else:
        train_loader, val_loader = oct2.get_oct2_dataloaders_sub(
            c_dataset=opt.dataset, need_idx=need_idx, raw=raw
        )
    # print(len(val_loader))

    model_s = model_dict[opt.model_s](num_classes=opt.n_classes, input_channel=1).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # print(list(torch.load(opt.path_s)["model"].keys())[320:])
    save_s = torch.load(opt.path_s)["model"]
    if STL:
        for _ in range(320):
            save_s.popitem(last=False)
        # _ = [print(i, n) for i, n in enumerate(save_s)]
        # exit()

        for i in list(save_s.keys()):
            save_s[i[8:]] = save_s[i]
            del save_s[i]
            # break

    # _ = [print(i, n) for i, n in enumerate(save_s)]
    # exit()
    model_s.load_state_dict(save_s)
    if need_idx:
        target, output, path_list = validate(
            val_loader=val_loader, model=model_s, need_idx=need_idx
        )
    else:
        target, output = validate(
            val_loader=val_loader, model=model_s, need_idx=need_idx
        )
        path_list = None
    print(f"tgt_len:{len(target)}, output_len:{len(output)}")
    print(">>> Confusion Matrix:")
    print(confusion_matrix(output, target))
    target = torch.tensor(target)
    output = torch.tensor(output)
    specificity = Specificity(num_classes=opt.n_classes)
    print(">>> Specificity:")
    print(specificity(output, target))
    print(">>> Sensitivity:")
    print(recall(output, target))
    print(">>> AUC:")
    print(auc(output, target, reorder=True))
    target = target.numpy()
    output = output.numpy()
    if need_idx:
        import pandas as pd

        print(len(target), len(output), len(path_list))
        df = pd.DataFrame(
            list(zip(target, output, path_list)), columns=["Label", "Output", "Path"]
        )
        print(df)
        df.to_excel(file_name)


if __name__ == "__main__":
    main()
