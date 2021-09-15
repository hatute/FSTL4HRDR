import os
import socket
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import random
from pathlib import Path
'''
(mean:[0.2951538], std:[0.1537334])
'''


# def get_imgs_paths(path):
#     imgs_path = []
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith('jpeg') or file.endswith('jpg') or file.endswith('png'):
#                 imgs_path.append(os.path.join(root, file))
#     # imgs_path = [os.path.join(path, i) for i in imgs_path if i.endswith('jpeg') ]
#     # print(imgs_path[0])
#     assert len(imgs_path) > 0, 'imgs_path set is void'
#     return imgs_path


def get_and_split(tifs, percent, shuffle=True):
    if shuffle:
        random.seed(30)
        random.shuffle(tifs)
    len_tgt_imgs = len(tifs)
    dvid_idx = int(len_tgt_imgs * percent)
    # print(len(tifs[:dvid_idx]), len(tifs[dvid_idx:]))
    return tifs[:dvid_idx], tifs[dvid_idx:]


def get_data_folder(test_percent, shuffle=True):
    """
    return server-dependent path to store the data
    """

    data_root = '../sample/boe'

    tif_path = []
    for i in os.walk(data_root):
        if len(i[2]) > 0:
            for file in i[2]:
                tif_path.append(os.path.join(i[0], file))
    test_data, train_data = get_and_split(tif_path, test_percent, shuffle)

    return train_data, test_data


def get_mean_std(dataset, ratio=0.5):
    """
    Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


class BOE_Dataset(Dataset):
    def __init__(self, samples_path, img_transform, need_idx=False):
        super(BOE_Dataset, self).__init__()
        self.img_transform = img_transform
        self.samples_path = samples_path
        self.n_total_samples = len(self.samples_path)
        self.need_idx = need_idx

    def __len__(self):
        return self.n_total_samples

    def __getitem__(self, index):
        path = self.samples_path[index]
        classes = path.split(os.sep)[-4]
        # print(classes)
        img, target = None, None

        if 'AMD' in classes:
            img = Image.open(path)
            target = 0
        elif 'DME' in classes:
            img = Image.open(path)
            target = 1
        elif 'NORMAL' in classes:
            img = Image.open(path)
            target = 2
        else:
            print("Unknown class, Please check!")

        if self.img_transform:
            img = self.img_transform(img)

        target = torch.tensor(target)

        assert img is not None and target is not None, 'img and tgt is None'

        if self.need_idx:
            return img, target, torch.tensor(index)
        else:
            return img, target


def get_boe_dataloaders(batch_size=128, num_workers=4, need_idx=False, **kwargs):
    """
    """
    train_paths, test_paths = get_data_folder(test_percent=0.15)

    size = 224

    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((size, size), (0.7, 1)),
            transforms.ToTensor(),
            # transforms.Normalize((0.20138986), (0.22367947)),
        ])

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # transforms.Normalize((0.20138986), (0.22367947)),
        ])

    train_set = BOE_Dataset(train_paths, train_transform, need_idx=need_idx)
    test_set = BOE_Dataset(test_paths, test_transform)

    print(len(train_set), len(test_set))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # print(train_loader.dataset)
    # print(test_loader.dataset)
    assert len(train_set) > len(test_set)
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_boe_dataloaders()
