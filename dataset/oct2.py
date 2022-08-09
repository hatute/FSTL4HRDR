import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import KFold
import numpy as np

"""
(mean:[0.2951538], std:[0.1537334])
"""


def get_imgs_paths(path):
    imgs_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
                imgs_path.append(os.path.join(root, file))
    # imgs_path = [os.path.join(path, i) for i in imgs_path if i.endswith('jpeg') ]
    # print(imgs_path[0])
    assert len(imgs_path) > 0, "imgs_path set is void"
    return imgs_path


def get_and_split(root_path, percent, shuffle=False):
    dirs = [
        i for i in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, i))
    ]
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
    # print(len(part_A), len(part_B))

    return part_A, part_B


def get_data_folder(data_roots, test_percent, shuffle=False):
    test_data = []
    train_data = []
    for dr in data_roots:
        print(f"read data from {dr}...")
        test, train = get_and_split(dr, test_percent, shuffle)
        print(f"leng of test:{len(test)}, leng of train:{len(train)}")
        test_data.extend(test)
        train_data.extend(train)

    if len(test_data) == 0:
        return train_data

    return train_data, test_data


# def get_mean_std(dataset, ratio=0.1):
#     """
#     Get mean and std by sample ratio
#     """
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
#                                              shuffle=True, num_workers=10)
#     train = iter(dataloader).next()[0]
#     mean = np.mean(train.numpy(), axis=(0, 2, 3))
#     std = np.std(train.numpy(), axis=(0, 2, 3))
#     return mean, std


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

        if classes == "central_preprocessed":
            img = Image.open(path)
            target = 0
        elif classes == "excluded central_preprocessed":
            img = Image.open(path)
            target = 1
        elif classes == "extensive_preprocessed":
            img = Image.open(path)
            target = 2
        elif classes == "normal_preprocessed":
            img = Image.open(path)
            target = 3
        elif classes == "control_preprocessed":
            img = Image.open(path)
            target = 4
        else:
            print("Unknown class, Please check!")

        if self.img_transform:
            img = self.img_transform(img)

        target = torch.tensor(target)

        assert img is not None and target is not None, "img and tgt is None"

        return (img, target, path) if self.need_idx else (img, target)


def get_oct2_dataloaders(
    c_dataset,
    batch_size=128,
    num_workers=4,
    need_idx=False,
    shuffle=True,
    verbose=False,
):
    """ """
    datasets = {
        "zs": (
            "/data/local/siwei/workspace/FSTL4HRDR/data/preprocessed/Zeiss_preprocessed",
        ),
        "oct2": (
            "/data/local/siwei/workspace/FSTL4HRDR/data/preprocessed/OCT2_preprocessed",
        ),
        "hd": (
            "/data/local/siwei/workspace/FSTL4HRDR/data/preprocessed/Heidelberg9_preprocessed",
        ),
    }

    train_paths, test_paths = get_data_folder(
        data_roots=datasets[c_dataset], test_percent=0.15
    )

    size = 224

    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((size, size), (0.7, 1)),
            transforms.ToTensor(),
            # transforms.Normalize((0.2951538), (0.1537334)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # transforms.Normalize((0.2951538), (0.1537334)),
        ]
    )

    train_set = OCT2_Dataset(train_paths, train_transform, need_idx=need_idx)
    test_set = OCT2_Dataset(test_paths, test_transform, need_idx=need_idx)
    if verbose:
        print(f"len of train is {len(train_set)}")
        print(f"len of test is {len(test_set)}")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    # print(train_loader.dataset)
    # print(test_loader.dataset)
    assert len(train_set) > len(test_set)
    print(f"length of train:{len(train_set)}\nlength of test:{len(test_set)}")
    return train_loader, test_loader


def get_kfold_dataloader(
    k,
    c_dataset,
    batch_size=128,
    num_workers=4,
    need_idx=False,
    shuffle=True,
    verbose=False,
):
    kfold = KFold(n_splits=k, shuffle=True)

    size = 224

    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((size, size), (0.7, 1)),
            transforms.ToTensor(),
            # transforms.Normalize((0.2951538), (0.1537334)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # transforms.Normalize((0.2951538), (0.1537334)),
        ]
    )
    zs = "/data/local/siwei/workspace/FSTL4HRDR/data/preprocessed/Zeiss_preprocessed"
    hb = "/data/local/siwei/workspace/FSTL4HRDR/data/preprocessed/Heidelberg9_preprocessed"

    all_path = None
    path_kfold = None
    if c_dataset == "zs":
        all_path = get_data_folder(data_roots=(zs,), test_percent=0)
        path_kfold = kfold.split(all_path)
    elif c_dataset == "hb":
        all_path = get_data_folder(data_roots=(hb,), test_percent=0)
        path_kfold = kfold.split(all_path)

    dataloader_set = []
    for idx, (train_idx, test_idx) in enumerate(path_kfold):
        all_path = np.array(all_path)
        train_path = all_path[train_idx]
        test_path = all_path[test_idx]
        train_set = OCT2_Dataset(train_path, train_transform, need_idx=need_idx)
        test_set = OCT2_Dataset(test_path, test_transform, need_idx=need_idx)

        if verbose:
            print(f"train len:{len(train_set)}")
            print(f"test len: {len(test_set)}")
            # exit()
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        assert len(train_set) > len(test_set)
        print(
            f"fold {idx} train_len:{len(train_set)}\nfold {idx} test_len :{len(test_set)}\n"
        )
        dataloader_set.append([train_loader, test_loader])
    assert (len(dataloader_set)) == k
    return dataloader_set


def get_hybird_dataloaders(
    c_train,
    c_test,
    batch_size=128,
    num_workers=4,
    need_idx=False,
    shuffle=True,
    verbose=False,
):
    """ """
    zs = "/data/local/siwei/workspace/FSTL4HRDR/data/preprocessed/Zeiss_preprocessed"
    hb = "/data/local/siwei/workspace/FSTL4HRDR/data/preprocessed/Heidelberg9_preprocessed"

    assert c_train != c_test, "Choice for train and test should not be same"
    train_paths = None
    test_paths = None
    if c_train == "zs" and c_test == "hb":
        train_paths = get_data_folder(data_roots=(zs,), test_percent=0)
        test_paths = get_data_folder(data_roots=(hb,), test_percent=0)
    elif c_train == "hb" and c_test == "zs":
        train_paths = get_data_folder(data_roots=(hb,), test_percent=0)
        test_paths = get_data_folder(data_roots=(zs,), test_percent=0)
    else:
        print("dataset choice has problem.")

    size = 224

    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((size, size), (0.7, 1)),
            transforms.ToTensor(),
            # transforms.Normalize((0.2951538), (0.1537334)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # transforms.Normalize((0.2951538), (0.1537334)),
        ]
    )

    train_set = OCT2_Dataset(train_paths, train_transform, need_idx=need_idx)
    test_set = OCT2_Dataset(test_paths, test_transform, need_idx=need_idx)
    if verbose:
        print(f"len of train is {len(train_set)}")
        print(f"len of test is {len(test_set)}")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    # print(train_loader.dataset)
    # print(test_loader.dataset)
    # assert len(train_set) > len(test_set)
    print(f"length of train:{len(train_set)}\nlength of test:{len(test_set)}")
    return train_loader, test_loader


if __name__ == "__main__":
    # dataset = datasets.ImageFolder("../../sample/oct2_c5",
    #                                transforms.Compose(
    #                                    [transforms.Grayscale(), transforms.Resize((224, 224)), transforms.ToTensor()]))
    # print(get_mean_std(dataset, 1))

    # get_oct2_dataloaders(verbose=True)

    # get_kfold_dataloader(5, 'zs')

    get_hybird_dataloaders("zs", "hb")
