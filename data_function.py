from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
RandomSpike,
RandomBlur,
RandomSwap,
    HistogramStandardization,
    OneOf,
    Clamp,
    Compose,
    RandomGhosting,
)
from pathlib import Path
# ************************************************************# ************************************************************
fold_arch = '*.nii.gz'
aug = False
crop_or_pad_size = 64,64,64
REs = 2.5
# ************************************************************# ************************************************************
class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir_0, images_dir_1):

        self.subjects = []


        images_dir_0 = Path(images_dir_0)
        self.image_paths_0 = sorted(images_dir_0.glob(fold_arch))

        images_dir_1 = Path(images_dir_1)
        self.image_paths_1 = sorted(images_dir_1.glob(fold_arch))

        for (image_path) in zip(self.image_paths_0):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label= 0,
            )
            self.subjects.append(subject)

        for (image_path) in zip(self.image_paths_1):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label= 1,
            )
            self.subjects.append(subject)




        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)


        # one_subject = self.training_set[0]
        # one_subject.plot()

    def transform(self):


        if aug: #数据增强开关
            training_transform = Compose([
            # CropOrPad((crop_or_pad_size), padding_mode='reflect'),#切割跟扩充
            # ToCanonical(),#规范坐标轴
            # RandomBiasField(),#随机 MRI 偏置场伪影

            ZNormalization(),#正则化
            Clamp(out_min=0, out_max=1000),
            Resample(REs),
            CropOrPad((crop_or_pad_size), padding_mode='reflect'),#切割跟扩充
            # RandomNoise(),#高斯噪声

            # RandomGhosting(),#MRI 鬼影伪影
            # RandomMotion(),#运动伪影
            # RandomSpike(), #MRI 尖峰伪影
            # RandomBlur(),#随机大小的高斯滤波器模糊图像
            # RandomSwap(),#随机交换图像中的补丁


            RandomFlip(axes=(0,)),#反转图像中元素的顺序
            # RandomAffine(),

            # OneOf({
            #     # RandomAffine(): 0.3,#倾斜
            #     # RandomElasticDeformation(): 0.1,#随机弹性变形
            # }),
            ])
        else:
            training_transform = Compose([
                ZNormalization(),  # 正则化
                Clamp(out_min=0, out_max=1000),
                Resample(REs),
                CropOrPad((crop_or_pad_size), padding_mode='reflect'),
            ])            


        return training_transform



# ************************************************************# ************************************************************
class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir_0, images_dir_1):

        self.subjects = []


        images_dir_0 = Path(images_dir_0)
        self.image_paths_0 = sorted(images_dir_0.glob(fold_arch))


        images_dir_1 = Path(images_dir_1)
        self.image_paths_1 = sorted(images_dir_1.glob(fold_arch))


        for (image_path) in zip(self.image_paths_0):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label= 0,
            )
            self.subjects.append(subject)

        for (image_path) in zip(self.image_paths_1):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label= 1,
            )
            self.subjects.append(subject)


        self.transforms = self.transform()

        self.testing_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)


        # one_subject = self.training_set[0]
        # one_subject.plot()

    def transform(self):


        testing_transform = Compose([

        ZNormalization(),  # 正则化
        Clamp(out_min=0, out_max=1000),
        Resample(REs),
        CropOrPad((crop_or_pad_size), padding_mode='reflect'),
        ])


        return testing_transform


# ************************************************************# ************************************************************

class MedData_val(torch.utils.data.Dataset):
    def __init__(self, val_0,val_1):
        self.subjects = []

        val_1 = Path(val_1)
        self.val_1 = sorted(val_1.glob(fold_arch))

        val_0 = Path(val_0)
        self.val_0 = sorted(val_0.glob(fold_arch))

        for (image_path) in zip(self.val_0):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=0,
                )
            self.subjects.append(subject)

        for (image_path) in zip(self.val_1):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=1,
                )
            self.subjects.append(subject)


        self.transforms = self.transform()

        self.testing_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

    def transform(self):

        val_transform = Compose([

        ZNormalization(),  # 正则化
        Clamp(out_min=0, out_max=1000),
        Resample(REs),
        CropOrPad((crop_or_pad_size), padding_mode='reflect'),
        ])

        return val_transform
# ************************************************************# ************************************************************
