import os
import socket
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import random

'''
(mean:[0.2951538], std:[0.1537334])
'''


def get_imgs_paths(path):
    imgs_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('jpeg') or file.endswith('jpg') or file.endswith('png'):
                imgs_path.append(os.path.join(root, file))
    # imgs_path = [os.path.join(path, i) for i in imgs_path if i.endswith('jpeg') ]
    # print(imgs_path[0])
    assert len(imgs_path) > 0, 'imgs_path set is void'
    return imgs_path


def get_and_split(root_path, percent, shuffle=True):
    dirs = [i for i in os.listdir(root_path) if os.path.isdir(
        os.path.join(root_path, i))]
    part_A = []
    part_B = []
    if shuffle:
        random.seed(30)
        random.shuffle(dirs)
    for i in dirs:
        tgt_path = os.path.join(root_path, i)
        tgt_imgs = get_imgs_paths(tgt_path)
        len_tgt_imgs = len(tgt_imgs)
        dvid_idx = int(len_tgt_imgs * percent)
        part_A.extend(tgt_imgs[:dvid_idx])
        part_B.extend(tgt_imgs[dvid_idx:])
    print(len(part_A), len(part_B))
    return part_A, part_B


def get_data_folder(test_percent, shuffle=True):
    """
    return server-dependent path to store the data
    """
    # hostname = socket.gethostname()
    # if hostname.startswith('CUDA1'):
    #     data_folder = '../../sample/kaggle'
    # elif hostname.startswith('Swells-MBP.local'):
    #     data_folder = '/home/yonglong/Data/data'
    # else:
    #     data_folder = './data/'
    #
    # if not os.path.isdir(data_folder):
    #     os.makedirs(data_folder)
    data_root = '../sample/oct2_c5'
    test_data, train_data = get_and_split(data_root, test_percent, shuffle)

    # train_folder = os.path.join(data_folder, "train_preprocessed")
    # test_folder = os.path.join(data_folder, "test_preprocessed")
    # return train_folder, test_folder

    return train_data, test_data


def get_mean_std(dataset, ratio=0.1):
    """
    Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


class OCT2_Dataset(Dataset):
    def __init__(self, samples_path, img_transform, need_idx=False):
        super(OCT2_Dataset, self).__init__()
        self.img_transform = img_transform
        self.samples_path = samples_path
        self.n_total_samples = len(self.samples_path)
        self.need_idx = need_idx

    def __len__(self):
        return self.n_total_samples

    def __getitem__(self, index):
        path = self.samples_path[index]
        classes = path.split(os.sep)[-2]

        img, target = None, None

        if 'central_preprocessed' == classes:
            img = Image.open(path)
            target = 0
        elif 'excluded central_preprocessed' == classes:
            img = Image.open(path)
            target = 1
        elif 'extensive_preprocessed' == classes:
            img = Image.open(path)
            target = 2
        elif 'normal_preprocessed' == classes:
            img = Image.open(path)
            target = 3
        elif 'control_preprocessed' == classes:
            img = Image.open(path)
            target = 4
        else:
            print("Unknown class, Please check!")

        if self.img_transform:
            img = self.img_transform(img)

        target = torch.tensor(target)

        assert img is not None and target is not None, 'img and tgt is None'

        if self.need_idx:
            return img, target, path
        else:
            return img, target


def get_oct2_dataloaders(batch_size=128, num_workers=4, need_idx=False, **kwargs):
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
            transforms.Normalize((0.2951538), (0.1537334)),
        ])

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.2951538), (0.1537334)),
        ])

    train_set = OCT2_Dataset(train_paths, train_transform, need_idx=need_idx)
    test_set = OCT2_Dataset(test_paths, test_transform, need_idx=need_idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # print(train_loader.dataset)
    # print(test_loader.dataset)
    assert len(train_set) > len(test_set)
    return train_loader, test_loader


if __name__ == '__main__':
    # dataset = datasets.ImageFolder("../../sample/oct2_c5",
    #                                transforms.Compose(
    #                                    [transforms.Grayscale(), transforms.Resize((224, 224)), transforms.ToTensor()]))
    # print(get_mean_std(dataset, 1))
    get_oct2_dataloaders()
