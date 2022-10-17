# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from .paco import PaCoLoss
from ..losses import accuracy
from .lovasz_loss import LovaszLoss


@HEADS.register_module()
class UPerHead_paco(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead_paco, self).__init__(
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

        # paco loss
        self.mlp = nn.Sequential(
             ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,inplace=False),
             ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,inplace=False),
             nn.Conv2d(self.channels, 128, 1))
        self.alpha = float(kwargs.get('alpha'))
        self.temperature = float(kwargs.get('temperature'))
        self.K = int(kwargs.get('K'))
        self.paco_loss = PaCoLoss(alpha=self.alpha, num_classes=self.num_classes, temperature=self.temperature)


    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
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
        seg_logit = self.cls_seg(output)
        embed = self.mlp(output)
        if self.training:
            return seg_logit, embed 
        else:
            return seg_logit


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logit, embed = self.forward(inputs)
        losses = self.losses(seg_logit, gt_semantic_seg)

        # reduced_label
        n, c, h, w = embed.shape
        reduced_seg_label = F.interpolate(gt_semantic_seg.to(torch.float32), size=(h, w), mode='nearest')
        reduced_seg_label = reduced_seg_label.long()

        # paco loss
        loss_paco = []
        for i in range(n):
            embed_s = embed[i].flatten(1).transpose(0,1).contiguous().view(-1, c)
            embed_s = F.normalize(embed_s, dim=1)
            seg_logit_t = seg_logit[i].flatten(1).transpose(0,1).contiguous().view(-1, self.num_classes)
            seg_label = torch.where(reduced_seg_label[i]>=self.num_classes, self.num_classes, reduced_seg_label[i])
            seg_label = seg_label.view(-1,)
            t = embed_s.size(0) if self.K == -1 else self.K 
            sample_index = torch.randperm(embed_s.size(0))[:t]
            loss_paco.append(self.paco_loss(embed_s[sample_index], seg_label[sample_index], seg_logit_t[sample_index]))
        losses['paco_loss'] = sum(loss_paco) / n
        return losses



@HEADS.register_module()
class UPerHead_paco11(UPerHead_paco):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead_paco11, self).__init__(pool_scales, **kwargs)
        self.lovasz_loss = LovaszLoss()
        self.lovasz_w = float(kwargs.get('lovasz_w'))
        self.paco_w = float(kwargs.get('paco_w'))


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logit, embed = self.forward(inputs)
        losses = self.losses(seg_logit, gt_semantic_seg)

        # reduced_label
        n, c, h, w = embed.shape
        reduced_seg_label = F.interpolate(gt_semantic_seg.to(torch.float32), size=(h, w), mode='nearest')
        reduced_seg_label = reduced_seg_label.long()

        # paco loss
        loss_paco = []
        for i in range(n):
            embed_s = embed[i].flatten(1).transpose(0,1).contiguous().view(-1, c)
            embed_s = F.normalize(embed_s, dim=1)
            seg_logit_t = seg_logit[i].flatten(1).transpose(0,1).contiguous().view(-1, self.num_classes)
            seg_label = torch.where(reduced_seg_label[i]>=self.num_classes, self.num_classes, reduced_seg_label[i])
            seg_label = seg_label.view(-1,)
            t = embed_s.size(0) if self.K == -1 else self.K
            sample_index = torch.randperm(embed_s.size(0))[:t]
            loss_paco.append(self.paco_loss(embed_s[sample_index], seg_label[sample_index], seg_logit_t[sample_index]))
        losses['paco_loss'] = self.paco_w * sum(loss_paco) / n

        # lovasz softmax
        seg_logit_upsample = resize(input=seg_logit, size=gt_semantic_seg.shape[2:],mode='bilinear',align_corners=self.align_corners)
        losses['loss_lovasz'] = self.lovasz_w * self.lovasz_loss(seg_logit_upsample, gt_semantic_seg.squeeze(1))
        return losses


@HEADS.register_module()
class UPerHead_paco_rr(UPerHead_paco):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead_paco_rr, self).__init__(pool_scales, **kwargs)
        self.paco_w = float(kwargs.get('paco_w'))
        self.rr_w = float(kwargs.get('rr_w'))
        frequency_file = kwargs.get('freq_file')
        self.smooth = float(kwargs.get('smooth', 1.0))

        print(self.smooth, ' smooth *********************')

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
            self.img_cls = nn.Sequential(
                 self.dropout,
                 nn.Linear(512, self.num_classes))
            for param in self.reduce.parameters():
                param.requires_grad = False
            for param in self.gain.parameters():
                param.requires_grad = False
            for param in self.img_cls.parameters():
                param.requires_grad = False


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
        seg_logit = self.cls_seg(output)

        # paco
        embed_paco = self.mlp(output)

        # rr
        if self.training:
           h, w = seg_label.shape[2:]
           pixel_features = self.reduce(output)
           pixel_features = F.interpolate(pixel_features, size=(h, w), mode='bilinear', align_corners=True)
           samples = []
           labels = []
           pixel_features = pixel_features.permute(0,2,3,1)

           y = seg_label.squeeze(1)
           _classes = torch.unique(y)
           for cls_index in _classes:
               tmp = pixel_features[y == cls_index,:]
               samples.append(tmp.mean(dim=0))
               labels.append(cls_index)

           ## img_level classification
           features = torch.stack(samples, dim=0)
           labels_img = torch.LongTensor(labels).cuda()
           img_x = self.gain(features)
           logits_img = self.img_cls(img_x)
           return seg_logit, logits_img, labels_img, embed_paco
        else:
           return seg_logit 


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logit, logits_img, labels_img, embed = self.forward(inputs, seg_label=gt_semantic_seg)
        losses = self.losses(seg_logit, gt_semantic_seg)

        # reduced_label
        n, c, h, w = embed.shape
        reduced_seg_label = F.interpolate(gt_semantic_seg.to(torch.float32), size=(h, w), mode='nearest')
        reduced_seg_label = reduced_seg_label.long()

        # paco loss
        loss_paco = []
        for i in range(n):
            embed_s = embed[i].flatten(1).transpose(0,1).contiguous().view(-1, c)
            embed_s = F.normalize(embed_s, dim=1)
            seg_logit_t = seg_logit[i].flatten(1).transpose(0,1).contiguous().view(-1, self.num_classes)
            seg_label = torch.where(reduced_seg_label[i]>=self.num_classes, self.num_classes, reduced_seg_label[i])
            seg_label = seg_label.view(-1,)
            t = embed_s.size(0) if self.K == -1 else self.K
            sample_index = torch.randperm(embed_s.size(0))[:t]
            loss_paco.append(self.paco_loss(embed_s[sample_index], seg_label[sample_index], seg_logit_t[sample_index]))
        losses['paco_loss'] = self.paco_w * sum(loss_paco) / n

        # rr loss
        labels_img_t = torch.where(labels_img >= self.num_classes, self.num_classes, labels_img)
        img_onehot = F.one_hot(labels_img_t, num_classes=self.num_classes+1)[:,:-1]
        img_onehot = self.smooth * img_onehot + (1 - self.smooth) / (self.num_classes - 1) * (1 - img_onehot)
        losses['loss_rr'] = -self.rr_w * (img_onehot * F.log_softmax(logits_img + torch.log(self.weight + 1e-12), dim=1)).sum() / (img_onehot.sum() + 1e-12)
        return losses


class UPerHead_rr(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead_rebalance, self).__init__(
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

            self.img_cls = nn.Sequential(
                 self.dropout,
                 nn.Linear(512, self.num_classes))

            for param in self.reduce.parameters():
                param.requires_grad = False
            for param in self.gain.parameters():
                param.requires_grad = False
            for param in self.img_cls.parameters():
                param.requires_grad = False


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


        # rr 
        if self.training:
           h, w = seg_label.shape[2:]
           pixel_features = self.reduce(output)
           pixel_features = F.interpolate(pixel_features, size=(h, w), mode='bilinear', align_corners=True)
           samples = []
           labels = []
           pixel_features = pixel_features.permute(0,2,3,1)

           y = seg_label.squeeze(1)
           _classes = torch.unique(y)
           for cls_index in _classes:
               tmp = pixel_features[y == cls_index,:]
               samples.append(tmp.mean(dim=0))
               labels.append(cls_index)

           ## img_level classification
           features = torch.stack(samples, dim=0)
           labels_img = torch.LongTensor(labels).cuda()
           img_x = self.gain(features)
           logits_img = self.img_cls(img_x)
           return final_output, logits_img, labels_img
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
        seg_logits, logits_img, labels_img = self.forward(inputs, gt_semantic_seg)
        losses = self.losses(seg_logits, gt_semantic_seg)

        # rr 
        labels_img_t = torch.where(labels_img >= self.num_classes, self.num_classes, labels_img)
        img_onehot = F.one_hot(labels_img_t, num_classes=self.num_classes+1)[:,:-1]
        img_onehot = self.smooth * img_onehot + (1 - self.smooth) / (self.num_classes - 1) * (1 - img_onehot)
        loss['loss_rr'] = -self.img_cls_weight * (img_onehot * F.log_softmax(logits_img + torch.log(self.weight + 1e-12), dim=1)).sum() / (img_onehot.sum() + 1e-12)
        return losses
