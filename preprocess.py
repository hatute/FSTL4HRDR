# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from skimage import io,  filters, transform
import random
import os
import argparse
import cv2
from skimage.util import crop
from matplotlib import pyplot as plt
import copy
import skimage.morphology as sm
from skimage.morphology import disk
import numpy as np
import bm3d
import glob
from tqdm import tqdm


# warnings.filterwarnings('error')
# np.seterr(all='ignore')


def crops(trans, mor):
    white = np.max(mor)
    white_area = np.argwhere(mor == white)
    height_white_area = white_area[:, 0]
    upside = np.min(height_white_area)
    downside = np.max(height_white_area)
    return crop(trans, ((upside, len(trans)-downside), (0, 0)))


def mid_bottom_line(img):
    white = np.max(img)
    mid_r = []
    bottom_r = []
    for i in range(len(img[0])):
        area = np.argwhere(img[:, i] == white)
        if len(area) == 0:
            mid_r.append(None)
            bottom_r.append(None)
        else:
            up = np.min(area)
            bottom = np.max(area)
            mid = (up+bottom) >> 1
            mid_r.append(mid)
            bottom_r.append(bottom)
    return mid_r, bottom_r


def method_judegement(mid_upwards, bot_upwards, p2_coe_mid, p1_coe_mid, p2_coe_bot, p1_coe_bot):
    use_p2 = None
    # print(mid_upwards)
    if mid_upwards:
        if p2_coe_mid >= p1_coe_mid:
            # p2+mid
            use_p2 = True
        else:
            # p1+mid
            use_p2 = False
    else:
        if bot_upwards:
            if p2_coe_bot >= p1_coe_bot:
                # p2+bot
                use_p2 = True
            else:
                # p1+bot
                use_p2 = False
        else:
            # p1+bot
            use_p2 = False
    return use_p2


def p2_alignment(p2_fit, trans, mor):
    avg_hook = np.average(p2_fit)
    diff_mov = p2_fit-avg_hook
    for i in range(len(trans[0])):
        diff = int(diff_mov[i])
        if diff != 0:
            mor[:, i] = np.array(mor[diff:, i].tolist()+mor[:diff, i].tolist())
            trans[:, i] = np.array(
                trans[diff:, i].tolist()+trans[:diff, i].tolist())
    return trans, mor


def p1_alignment(p1_args, img, mor):
    degree = np.degrees(np.arctan2(p1_args[0], 1))
    rotated = transform.rotate(img, degree, preserve_range=True)
    mor = transform.rotate(mor, degree, preserve_range=True)
    return rotated, mor


def alignment(use_p2, p2_fit, p1_args, img, mor):
    if use_p2:
        return p2_alignment(p2_fit, img, mor)
    else:
        return p1_alignment(p1_args, img, mor)


def fill_black(src_img):
    img = copy.deepcopy(src_img)
    white = 1.0 if np.max(img) <= 1 else 255
    white_area = np.argwhere(img >= white*0.96)
    for i in white_area:
        img[i[0], i[1]] = 0
    return img


def path_dealer(path_root):

    if path_root.split('/')[-1] == '':
        golbed_path = path_root+'**/*.jpeg'
        path_root = path_root[:-1]
    else:
        golbed_path = path_root+'/**/*.jpeg'

    total_path = glob.glob(golbed_path, recursive=True)
    new_root = path_root+'_preprocessed'
    return total_path, new_root


def preprocess_single(src_img_path, new_root, need_save=True, skip_dul=True):
    try:
        splited = os.path.split(src_img_path)[0].split('/')[-1]+'_preprocessed'
        output_path = os.path.join(new_root, splited)
        tgt_img_name = 'preprocessed_'+src_img_path.split('/')[-1]
        tgt_name = os.path.join(output_path, tgt_img_name)
        if os.path.exists(tgt_name) and skip_dul:
            return
        # if not os.path.isdir(output_path):
        #     os.makedirs(output_path)
        read_img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
        src_img = fill_black(read_img)
        denoised_img_ALL_01 = bm3d.bm3d(
            src_img, sigma_psd=0.1, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING).astype('uint8')
        bi_val = filters.threshold_otsu(
            denoised_img_ALL_01).astype('uint8')
        otsued = np.digitize(denoised_img_ALL_01, bins=[
            bi_val]).astype('uint8')
        median_filtered = filters.median(otsued, disk(5))
        closed = sm.closing(median_filtered, disk(30))
        opened = sm.opening(closed, disk(3))

        raw_mid_line, raw_bottom_line = mid_bottom_line(opened)

        x_ranges = [i for i in range(len(raw_mid_line))
                    if raw_mid_line[i] is not None]
        mid_line = [i for i in (raw_mid_line) if i is not None]
        bottom_line = [i for i in (raw_bottom_line) if i is not None]

        poly1_args_mid = np.polyfit(x_ranges, mid_line, 1)
        poly1_args_bot = np.polyfit(x_ranges, bottom_line, 1)

        poly2_args_mid = np.polyfit(x_ranges, mid_line, 2)
        poly2_args_bot = np.polyfit(x_ranges, bottom_line, 2)

        mid_upwards = poly2_args_mid[0] < 0
        bot_upwards = poly2_args_bot[0] < 0

        p1_coe_mid = np.corrcoef(mid_line, np.poly1d(
            poly1_args_mid)(x_ranges))[0][1]

        p1_coe_bot = np.corrcoef(bottom_line, np.poly1d(
            poly1_args_bot)(x_ranges))[0][1]

        p2_coe_mid = np.corrcoef(mid_line, np.poly1d(
            poly2_args_mid)(x_ranges))[0][1]
        p2_coe_bot = np.corrcoef(bottom_line, np.poly1d(
            poly2_args_bot)(x_ranges))[0][1]

        methods = method_judegement(
            mid_upwards, bot_upwards, p2_coe_mid, p1_coe_mid, p2_coe_bot, p1_coe_bot)

        p2_fit = np.poly1d(poly2_args_mid)(np.arange(0, len(read_img[0])))

        trans, mask = alignment(
            methods, p2_fit, poly1_args_mid, read_img, opened)
        cropped = crops(trans, mask).astype('uint8')

        if need_save:
            io.imsave(tgt_name, cropped)

    except (Exception, Warning) as e:
        print('*'*20)
        print(e)
        print(src_img_path)
        print('*'*20)
        print()


def initialize(root_path):
    total_path, new_root = path_dealer(root_path)

    print("~"*40)
    print("detected image #: {}".format(len(total_path)))
    print('target root path: \"{}\"'.format(new_root))

    cat = {}
    for i in total_path:
        tgt = i.split('/')[-2]
        if tgt not in cat:
            cat[tgt] = 1
        else:
            cat[tgt] += 1
    print('categories: {}'.format(cat))

    for i in cat.keys():
        tgt_path = os.path.join(new_root, i+'_preprocessed')
        # print(tgt_path)
        if os.path.exists(tgt_path):
            print(tgt_path+" is existed")
        else:
            os.makedirs(tgt_path)
            print(tgt_path+" is created")
    print("~"*40)
    return total_path, new_root


parser = argparse.ArgumentParser()
parser.add_argument("path", help="target root path")
args = parser.parse_args()
root_path = args.root_path

total_path, new_root = initialize(root_path)
random.shuffle(total_path)
print("run in shuffled mode!")
for i in tqdm((range(len(total_path)))):
    preprocess_single(total_path[i], new_root, need_save=True, skip_dul=True)

print('Done!')
