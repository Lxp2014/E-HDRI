# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
from models.LRecModel import LRecModel
import util.util as util
import PIL.Image as img
from PIL import Image
import pytorch_ssim
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity
import cv2
import glob as gb
from os.path import join
from torchstat import stat
from thop import profile
import time

def saturated_channel_yuhu(im, th):
    return np.minimum(im / (1 - th), np.minimum(1, (1 - im) / (1 - th)))


def get_saturated_regions_yuhu(im, th=0.7):
    w, h = im.shape
    mask_con = np.zeros_like(im)
    mask_con[:, :] = saturated_channel_yuhu(im[:, :], th)
    return mask_con


def crop_256(img):
    img = img[:,52:308,112:368]
    return img

def mask_1(img_yuv):
    yy = np.array(img_yuv)
    yy = yy.astype(float) / 255.0
    mask = get_saturated_regions_yuhu(yy)
    return mask

def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.checkpoints_dir = "./checkpoints/"

if __name__ == "__main__":

    opt = TestOptions().parse(save=False)
    parameter_set(opt)
    print("*************************************pth:", opt.which_epoch)
    model = LRecModel()
    model.initialize(opt)
    model.eval()

    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
    dataset_size = 0

    dataroot = opt.test_input
    LDRimages = gb.glob(os.path.join(dataroot, '*/event/*.npy'))
    # print('LDRimages',LDRimages)
    LDRimages.sort()

    transform_list_L = []
    transform_list_L.append(transforms.ToTensor())
    # transform_list_L.append(transforms.Normalize((0.5,), (0.5,)))
    transform_img_L = transforms.Compose(transform_list_L)

    transform_list_mask = []
    transform_list_mask.append(transforms.ToTensor())
    transform_mask = transforms.Compose(transform_list_mask)



    psnr_sum = 0
    ssim_sum = 0
    sum_time = 0
    error_list = []
    for i in range(len(LDRimages)):
        if int(LDRimages[i].split('/')[-1].split('.')[0])%1==0:
            LDR_path = LDRimages[i].replace('event','LDR').replace('.npy','.png')
            eventpth = LDR_path.replace('LDR','event').replace('.png','.npy')

            LDR_BGR = cv2.imread(LDR_path)
            LDR_BGR = cv2.resize(LDR_BGR,(480,360))
            cv2.imwrite(LDR_path, LDR_BGR)
            LDR_RGB = cv2.cvtColor(LDR_BGR, cv2.COLOR_BGR2RGB)
            LDR_data = transform_img_L(LDR_RGB)

            event_data = np.load(eventpth, allow_pickle=True)
            event_data = torch.from_numpy(event_data)

            img_LDR_mask = mask_1(cv2.cvtColor(LDR_BGR, cv2.COLOR_BGR2GRAY))
            img_LDR_mask = transform_mask(img_LDR_mask)

            # LDR_data = crop_256(LDR_data).unsqueeze(0)
            # event_data = crop_256(event_data).unsqueeze(0)
            # img_LDR_mask = crop_256(img_LDR_mask).unsqueeze(0)
            LDR_data = torch.nn.functional.interpolate(LDR_data.unsqueeze(0), (384, 640), mode='bilinear')
            event_data = torch.nn.functional.interpolate(event_data.unsqueeze(0), (384, 640), mode='bilinear')
            img_LDR_mask = torch.nn.functional.interpolate(img_LDR_mask.unsqueeze(0), (384, 640), mode='bilinear')
            
            a = time.time()
            generated, mask = model.inference(LDR_data, event_data,img_LDR_mask)
            b = time.time()
            sum_time = sum_time + b-a
            total = sum([param.nelement() for param in model.parameters()])
            print('Number of parameter: %.2fM' % (total/1e6))

            gene_nump = generated.data.cpu().numpy()

            gene_nump2 = gene_nump[0,:,:,:]
            gene_nump3 = gene_nump2.transpose(1, 2, 0)
            gene_nump4 = gene_nump3[:, :, ::-1]
            gene_nump4 = tonemapReinhard.process(gene_nump4)
            gene_nump4 = (gene_nump4.clip(0, 1) * 255).astype(np.uint8)
            outputs_dir = os.path.join(LDR_path.split('LDR')[0], 'ours')
            ensure_dir(outputs_dir)
            gene_nump4 = cv2.resize(gene_nump4,(480,360))
            cv2.imwrite(LDR_path.replace('LDR','ours'), gene_nump4)
            # cv2.imwrite(join(outputs_dir, 'test_static',name_LDR+'.jpg'), (gene_nump4 * 255).astype(int))
    print('ave_time', sum_time/len(LDRimages))
