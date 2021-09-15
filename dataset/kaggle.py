import os
import socket
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

'''
(mean:[0.28563553], std:[0.17276825])
'''


def get_data_folder():
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
    data_folder = '../sample/kaggle_c4'
    train_folder = os.path.join(data_folder, "train_preprocessed")
    test_folder = os.path.join(data_folder, "test_preprocessed")
    return train_folder, test_folder


def get_mean_std(dataset, ratio=0.01):
    """
    Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]  # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


def get_kaggle_dataloaders(batch_size=128, num_workers=4):
    """
    kaggle
    Train:CNV(36872),DME(11251), DRUSEN(8531), NORMAL(50750)
    Test: 250/class
    """
    train_folder, test_folder = get_data_folder()

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.RandomResizedCrop((224, 224), (0.65, 1)),
            transforms.ToTensor(),
            transforms.Normalize((0.28563553), (0.17276825)),
        ])

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.28563553), (0.17276825)),
        ])

    train_set = datasets.ImageFolder(train_folder, train_transform)
    test_set = datasets.ImageFolder(test_folder, test_transform)
    '''
    {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}
    '''
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # print(train_loader.dataset)
    # print(test_loader.dataset)
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_kaggle_dataloaders()

    # print(get_mean_std(dataset))
