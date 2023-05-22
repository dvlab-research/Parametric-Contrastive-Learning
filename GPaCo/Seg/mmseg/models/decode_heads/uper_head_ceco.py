# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from ..losses import accuracy
import random
from scipy.stats import ortho_group

@HEADS.register_module()
class UPerHead_cecol(BaseDecodeHead):  # 'Linear' version of CeCo, i.e., RR 
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead_cecol, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        frequency_file = kwargs.get('frequency_file')
        self.img_cls_weight = float(kwargs.get('img_cls_weight'))
        self.smooth = float(kwargs.get('smooth', 1.0))
        self.etf = float(kwargs.get('etf', False))
        self.scale = float(kwargs.get('scale', 1.0))

        ## rebalance
        img_w_list = []
        content = open(frequency_file, "r").readlines()
        for line in content:
            img_w_list.append(int(line))

        self.weight = torch.Tensor(img_w_list)
        self.weight = self.weight / self.weight.sum()
        self.weight = self.weight.view(1,self.num_classes)
        self.weight = nn.parameter.Parameter(self.weight, requires_grad=False)

        if self.training:
            self.reduce = nn.Sequential(
                 nn.Conv2d(self.channels, 128, kernel_size=1, padding=0, bias=False),
                 nn.BatchNorm2d(128))
            self.gain = nn.Sequential(
                 nn.Linear(128, 512),
                 nn.ReLU(inplace=True))
            self.img_cls = nn.Sequential(self.dropout, nn.Linear(512, self.num_classes))

            for param in self.reduce.parameters():
                param.requires_grad = False
            for param in self.gain.parameters():
                param.requires_grad = False
            for param in self.img_cls.parameters():
                param.requires_grad = False

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)

        output = self.conv_seg(feat)
        return output

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs, seg_label=None):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        final_output = self.cls_seg(output)

        # decoupling
        if self.training:
           h, w = seg_label.shape[2:]
           pixel_features = self.reduce(output)
           pixel_features = F.interpolate(pixel_features, size=(h, w), mode='bilinear', align_corners=True)
           pixel_features = pixel_features.permute(0,2,3,1).contiguous()
           y = seg_label.squeeze(1)

           y_valid = y[y!=255].long().cuda()
           out_valid = pixel_features[y!=255,:]
           y_onehot = F.one_hot(y_valid, self.num_classes).float()
           features = y_onehot.T @ out_valid

           scene_label = torch.unique(y_valid)
           features = features[scene_label,:]
           cls_num = y_onehot.T.sum(dim=1)
           cls_num = cls_num[scene_label]
           features = features / cls_num.unsqueeze(1)

           img_x = self.gain(features)
           f = F.normalize(img_x, dim=-1) if self.etf else img_x
           logits_img = self.img_cls(f) * self.scale
           return final_output, seg_label, logits_img, scene_label
        else:
           return final_output


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        outputs = self.forward(inputs, gt_semantic_seg)
        losses = self.losses(outputs[0], outputs[1], outputs[2], outputs[3])
        return losses


    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, logits_img, labels_img):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        for loss_decode in self.loss_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        # rebalance loss for img
        labels_img_t = torch.where(labels_img >= self.num_classes, self.num_classes, labels_img)
        img_onehot = F.one_hot(labels_img_t, num_classes=self.num_classes+1)[:,:-1]
        img_onehot = self.smooth * img_onehot + (1 - self.smooth) / (self.num_classes - 1) * (1 - img_onehot)
        loss['loss_img_cls'] = -(img_onehot * F.log_softmax(logits_img + torch.log(self.weight + 1e-12), dim=1)).sum()  / (img_onehot.sum() + 1e-12) * self.img_cls_weight
        return loss


@HEADS.register_module()
class UPerHead_ceco(UPerHead_cecol):    # Normalized 
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead_ceco, self).__init__(pool_scales=pool_scales, **kwargs)

        orth = ortho_group.rvs(dim=512).astype(np.float32)
        orth = torch.tensor(orth[:,:self.num_classes])
        etf = math.sqrt(self.num_classes/(self.num_classes-1)) * orth @ (torch.eye(self.num_classes) - 1.0 / self.num_classes * torch.ones(self.num_classes, self.num_classes))
        etf = etf.t()
        self.img_cls[1].weight.data = etf / etf.norm(dim=-1, keepdim=True)
        self.img_cls[1].bias.data = torch.zeros(self.num_classes,)

        self.etf = True
