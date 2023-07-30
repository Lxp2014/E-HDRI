# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os.path
import io
import zipfile
import os.path as osp
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
from io import BytesIO
import glob
import torch
import json
def saturated_channel_yuhu(im, th):
    return np.minimum(im / (1 - th), np.minimum(1, (1 - im) / (1 - th)))


def get_saturated_regions_yuhu(im, th=0.7):
    w, h = im.shape
    mask_con = np.zeros_like(im)
    mask_con[:, :] = saturated_channel_yuhu(im[:, :], th)
    return mask_con


def mask(img_yuv):
    yy = np.array(img_yuv)
    yy = yy.astype(float) / 255.0
    mask = get_saturated_regions_yuhu(yy)
    return mask

class EventHDR_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        print(opt.name)
        self.dataroot = opt.dataroot
        self.event_number = opt.event_number
        self.crop_sz_H = 256
        self.crop_sz_W = 256

        ####################################################################################################
        transform_list_L = []
        transform_list_L.append(transforms.ToTensor())
        #transform_list_L.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform_img_L = transforms.Compose(transform_list_L)

        transform_list_mask = []
        transform_list_mask.append(transforms.ToTensor())
        self.transform_mask = transforms.Compose(transform_list_mask)
        ####################################################################################################
        self.scence = glob.glob(os.path.join(self.dataroot, '*'))
        self.imnames = []
        for scence_name in self.scence:
            image_list = np.load(os.path.join(scence_name, 'image_list.npy'))
            for i in range(len(image_list)):
                if int(image_list[i].split('.')[0])%4==2:  # if int(image_list[i].split('.')[0])%3==0:
                    self.imnames.append(os.path.join(scence_name, 'Master', 'LDR_align_8', image_list[i]))
        random.shuffle(self.imnames)

    def __getitem__(self, index):
        name = self.imnames[index]
        self.LDRpth = name.replace('.tif', '.png')
        self.HDRpth = name.replace('.tif', '.hdr').replace('LDR_align_8', 'HDR_align')
        self.event_PN_pth = name.replace('Master', 'event').replace('LDR_align_8', 'event_vox_np_15m').replace('.tif', '.npy')

        LDR_BGR = cv2.imread(self.LDRpth)
        LDR_RGB = cv2.cvtColor(LDR_BGR, cv2.COLOR_BGR2RGB)
        LDR_data = self.transform_img_L(LDR_RGB)

        HDR_BGR = cv2.imread(self.HDRpth, -1)
        HDR_RGB = cv2.cvtColor(HDR_BGR, cv2.COLOR_BGR2RGB)
        HDR_data = self.transform_img_L(HDR_RGB)

        # pdb.set_trace()
        event_data_PN = np.load(self.event_PN_pth, allow_pickle=True)
        event_data = torch.from_numpy(event_data_PN)

        img_LDR_mask = mask(cv2.cvtColor(LDR_BGR, cv2.COLOR_BGR2GRAY))
        img_LDR_mask = self.transform_mask(img_LDR_mask)

        y = np.random.randint(low=1, high=(LDR_data.shape[1] - self.crop_sz_H))
        x = np.random.randint(low=1, high=(LDR_data.shape[2] - self.crop_sz_W))
        event_data = event_data[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]
        HDR_data = HDR_data[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]
        LDR_data = LDR_data[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]
        img_LDR_mask = img_LDR_mask[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]

        input_dict = {'LDR': LDR_data,  'event_data': event_data, 'HDR': HDR_data, 'img_LDR_mask':img_LDR_mask}

        return input_dict

    def __len__(self):
        return len(self.imnames)  ## actually, this is useless, since the selected index is just a random number

    def name(self):
        return 'EventHDR_Dataset'
