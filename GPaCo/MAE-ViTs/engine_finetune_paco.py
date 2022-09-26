# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if isinstance(samples, list):
           is_paco = True
           samples_v1, samples_v2 = samples[0], samples[1]
           samples_v1 = samples_v1.to(device, non_blocking=True)
           samples_v2 = samples_v2.to(device, non_blocking=True)
        else:
           is_paco = False
           samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if not is_paco:
               features, labels, outputs = model(samples, y=targets)
            else:
               features, labels, outputs = model([samples_v1, samples_v2], y=targets)
            loss = criterion(features, labels, outputs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import PIL
from torchvision import datasets, transforms
import os
import torch.nn.functional as F
from util.imagenet_r import *


@torch.no_grad()
def test(net, test_loader, mark=None):
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      if mark == 'imagenet-r':
         logits = logits[:,imagenet_r_mask]
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)



CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]


def compute_mce(corruption_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  for i in range(len(CORRUPTIONS)):
    avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
    ce = 100 * avg_err / ALEXNET_ERR[i]
    mce += ce / 15
  return mce


@torch.no_grad()
def evaluate_c(net, args):

    # test_transform
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
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
    test_transform = transforms.Compose(t)


    # calculate
    corruption_accs = {}
    for c in CORRUPTIONS:
        print(c)
        for s in range(1, 6):
            valdir = os.path.join(args.corrupted_data, c, str(s))
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, test_transform),
                batch_size=100,
                shuffle=False,
                num_workers=24,
                pin_memory=True)

            loss, acc1 = test(net, val_loader)
            if c in corruption_accs:
                corruption_accs[c].append(acc1)
            else:
                corruption_accs[c] = [acc1]

            print('\ts={}: Test Loss {:.3f} | Test Acc1 {:.3f}'.format(s, loss, 100. * acc1))

    mce = 0
    for i in range(len(CORRUPTIONS)):
        avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
        ce = 100 * avg_err
        mce += ce / 15
    print('mce: ',mce) 
    rel_mce = compute_mce(corruption_accs)
    print('rel mce: ',rel_mce)


def evaluate_r(net, args):

    # test_transform
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
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
    test_transform = transforms.Compose(t)


    # calculate
    print('imagenet-r')
    valdir = args.imagenet_r_data
    val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, test_transform),
                batch_size=100,
                shuffle=False,
                num_workers=24,
                pin_memory=True)

    loss, acc1 = test(net, val_loader, mark='imagenet-r')
    print("loss: ",loss, 'acc1: \n',acc1)


def evaluate_s(net, args):

    # test_transform
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
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
    test_transform = transforms.Compose(t)


    # calculate
    print('imagenet-s')
    valdir = args.imagenet_s_data
    val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, test_transform),
                batch_size=100,
                shuffle=False,
                num_workers=24,
                pin_memory=True)

    loss, acc1 = test(net, val_loader, mark='imagenet-s')
    print("loss: ",loss, 'acc1: \n',acc1)
