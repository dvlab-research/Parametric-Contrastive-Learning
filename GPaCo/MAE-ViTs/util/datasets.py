# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import moco.loader
from randaugment import rand_augment_transform, GaussianBlur

def build_dataset(is_train, args, is_paco=False, singleview=False):
    if not is_paco:
        transform = build_transform(is_train, args)
    else:
        if not singleview:
            transform = build_transform_paco(is_train, args)
        else:
            transform = build_transform_singleview(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    transform = build_transform_paco(is_train, args)
    print(dataset)
    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_transform_paco(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)
        augmentation_randncls_stack = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.rand_n, args.rand_m), ra_params),
            transforms.ToTensor(),
            normalize,
        ]

        augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.rand_n, args.rand_m), ra_params),
            transforms.ToTensor(),
            normalize,
        ]

        transform = moco.loader.TwoCropsTransformV2([transforms.Compose(augmentation_randncls_stack), transforms.Compose(augmentation_sim)])
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_transform_singleview(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)
        augmentation_randncls_stack = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.rand_n, args.rand_m), ra_params),
            transforms.ToTensor(),
            normalize,
        ]

        augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.rand_n, args.rand_m), ra_params),
            transforms.ToTensor(),
            normalize,
        ]

        transform = transforms.Compose(augmentation_randncls_stack)
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


