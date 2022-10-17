import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from ..losses import accuracy
from .aspp_head import ASPPHead, ASPPModule
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .cascade_decode_head import BaseCascadeDecodeHead


@HEADS.register_module()
class UPerHead_regionrebalance(BaseDecodeHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead_regionrebalance, self).__init__(
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


        ## region cls seg
        self.region_seg_bn = nn.Sequential(
             nn.Conv2d(self.channels, self.channels, 1),
             nn.ReLU(inplace=True)
        )
        self.region_conv_seg = nn.Conv2d(self.channels, self.num_classes+1, kernel_size=1)

        ## rebalance
        frequency_file = kwargs.get('frequency_file')
        self.region_w = float(kwargs.get('region_w'))
        self.smooth = float(kwargs.get('smooth'))
        self.main_w = float(kwargs.get('main_w'))
        self.neg_w = float(kwargs.get('neg_w'))
        self.prior = bool(kwargs.get('prior'))

        img_w_list = []
        content = open(frequency_file, "r").readlines()
        for line in content:
            img_w_list.append(int(line))

        weight = torch.Tensor(img_w_list)
        weight = (weight / weight.sum()).view(1, self.num_classes)
        self.weight = torch.ones(1, self.num_classes+1)
        self.weight[:,:-1] = weight
        self.weight = nn.parameter.Parameter(self.weight, requires_grad=False)

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

        # region
        #final_output_t = final_output.permute(0,2,3,1)
        #score = F.gumbel_softmax(final_output_t, tau=1, hard=False).permute(0,3,1,2).flatten(2)
        score = F.softmax(final_output, dim=1).flatten(2)                  # n 150, hw 
        feats = output.flatten(2)                                          # n 512, hw
        res_w = (score @ feats.transpose(1,2)) / (score.sum(dim=2, keepdim=True) + 1e-12)  # n 150 hw, n hw 512 -> n 150 512
        n = output.size(0)
        region_bias = self.region_cls_seg(res_w.view(n * self.num_classes, self.channels, 1, 1))
        region_bias = region_bias.view(n, self.num_classes, self.num_classes + 1)
        res_output = (region_bias[:,:,:-1]).mean(1).view(n, self.num_classes, 1, 1)

        # decoupling
        if self.training:
           n, c, h, w = output.shape
           reduced_seg_label = F.interpolate(seg_label.to(torch.float32), size=(h, w), mode='nearest')
           reduced_seg_label = reduced_seg_label.squeeze(1).long()

           samples = []
           labels = []
           pixel_features = output.permute(0,2,3,1)
           y = reduced_seg_label.squeeze(1)
           _classes = torch.unique(y)
           for cls_index in _classes:
               tmp = pixel_features[y == cls_index,:]
               samples.append(tmp.mean(dim=0))
               labels.append(cls_index)

           ## region classification
           features = torch.stack(samples, dim=0).unsqueeze(2).unsqueeze(3)
           labels_img = torch.LongTensor(labels).cuda()
           logits_img = self.region_cls_seg(features)
           logits_img = logits_img.squeeze(3).squeeze(2)
           return final_output, seg_label, logits_img, labels_img, region_bias 
        else:
           return final_output + res_output

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
        losses = self.losses(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
        return losses

    def region_cls_seg(self, feat):
        feat = self.region_seg_bn(feat)
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.region_conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, logits_img, labels_img, region_bias):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit_upsample = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        all_logit_upsample = seg_logit_upsample + (region_bias[:,:,:-1]).mean(1).unsqueeze(2).unsqueeze(3)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        # pixel loss
        loss['acc_seg_seg'] = accuracy(seg_logit_upsample, seg_label)
        loss['acc_seg_all'] = accuracy(all_logit_upsample, seg_label)
        loss['loss_seg_seg'] = self.main_w * F.cross_entropy(seg_logit_upsample, seg_label, weight=seg_weight, ignore_index=self.ignore_index)
        loss['loss_seg_all'] = self.main_w * F.cross_entropy(all_logit_upsample, seg_label, weight=seg_weight, ignore_index=self.ignore_index)

        # img aware loss 
        mask = []
        seg_label_t = torch.where(seg_label==self.ignore_index, self.num_classes, seg_label)
        n, h, w = seg_label_t.shape
        region_p_gts = torch.ones(n, self.num_classes + 1, dtype=torch.long).cuda() * self.num_classes
        for i in range(n):
            _classes = torch.unique(seg_label_t[i])
            region_p_gts[i,_classes] = _classes
            img_onehot = F.one_hot(_classes, num_classes=self.num_classes+1).sum(dim=0)
            mask.append(img_onehot)
        region_p_gts = (region_p_gts[:,:-1]).contiguous()
        mask = torch.stack(mask, dim=0)[:,:-1].contiguous().view(n, self.num_classes, 1, 1)
        loss['acc_seg_seg_mask'] = accuracy(seg_logit_upsample * mask, seg_label)
        loss['acc_seg_all_mask'] = accuracy(all_logit_upsample * mask, seg_label)
        loss['loss_seg_all_mask'] = self.main_w * F.cross_entropy(all_logit_upsample * mask, seg_label, weight=seg_weight, ignore_index=self.ignore_index)

        # region_p cls
        mask = ((region_p_gts == self.num_classes) * self.neg_w + region_p_gts != self.num_classes).view(-1,1)
        region_bias = region_bias.view(n * self.num_classes, -1)
        region_p_onehot = F.one_hot(region_p_gts.view(-1,), num_classes=self.num_classes + 1)
        region_p_onehot = self.smooth * region_p_onehot + (1 - self.smooth) / self.num_classes * (1 - region_p_onehot)
        if self.prior:
           region_bias = region_bias + torch.log(self.weight + 1e-12)
        loss['loss_region_p_cls'] = -self.region_w * (mask * region_p_onehot * F.log_softmax(region_bias, dim=1)).sum() / (mask.sum() + 1e-12)

        # region_gt cls 
        region_gts_t = torch.where(labels_img >= self.num_classes, self.num_classes, labels_img)
        mask = ((region_gts_t == self.num_classes) * self.neg_w + region_gts_t != self.num_classes).view(-1,1)
        region_onehot = F.one_hot(region_gts_t, num_classes=self.num_classes + 1)
        region_onehot = self.smooth * region_onehot + (1 - self.smooth) / self.num_classes * (1 - region_onehot)
        if self.prior:
           logits_img = logits_img + torch.log(self.weight + 1e-12)
        loss['loss_img_cls'] = -self.region_w * (mask * region_onehot * F.log_softmax(logits_img, dim=1)).sum() / (mask.sum() + 1e-12)
        return loss



@HEADS.register_module()
class FCNHead_regionrebalance(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.
    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead_regionrebalance, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        ## region cls seg
        self.region_seg_bn = nn.Sequential(
             nn.Conv2d(self.channels, self.channels, 1),
             nn.ReLU(inplace=True))
        self.region_conv_seg = nn.Conv2d(self.channels, self.num_classes+1, kernel_size=1)

        ## rebalance
        frequency_file = kwargs.get('frequency_file')
        self.region_w = float(kwargs.get('region_w'))
        self.smooth = float(kwargs.get('smooth'))
        self.main_w = float(kwargs.get('main_w'))
        self.neg_w = float(kwargs.get('neg_w'))
        self.prior = bool(kwargs.get('prior'))

        img_w_list = []
        content = open(frequency_file, "r").readlines()
        for line in content:
            img_w_list.append(int(line))

        weight = torch.Tensor(img_w_list)
        weight = (weight / weight.sum()).view(1, self.num_classes)
        self.weight = torch.ones(1, self.num_classes+1)
        self.weight[:,:-1] = weight
        self.weight = nn.parameter.Parameter(self.weight, requires_grad=False)
        self.criterion_kl = nn.KLDivLoss(reduction='none')


    def forward(self, inputs, seg_label=None):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        final_output = self.cls_seg(output)

        # region
        #final_output_t = final_output.permute(0,2,3,1)
        #score = F.gumbel_softmax(final_output_t, tau=1, hard=False).permute(0,3,1,2).flatten(2)
        score = F.softmax(final_output, dim=1).flatten(2)                  # n 150, hw 
        feats = output.flatten(2)                                          # n 512, hw
        res_w = (score @ feats.transpose(1,2)) / (score.sum(dim=2, keepdim=True) + 1e-12)  # n 150 hw, n hw 512 -> n 150 512
        n = output.size(0)
        region_bias = self.region_cls_seg(res_w.view(n * self.num_classes, self.channels, 1, 1))
        region_bias = region_bias.view(n, self.num_classes, self.num_classes + 1)
        res_output = (region_bias[:,:,:-1]).mean(1).view(n, self.num_classes, 1, 1)

        # decoupling
        if self.training:
           n, c, h, w = output.shape
           reduced_seg_label = F.interpolate(seg_label.to(torch.float32), size=(h, w), mode='nearest')
           reduced_seg_label = reduced_seg_label.squeeze(1).long()

           samples = []
           labels = []
           pixel_features = output.permute(0,2,3,1)
           y = reduced_seg_label.squeeze(1)
           _classes = torch.unique(y)
           for cls_index in _classes:
               tmp = pixel_features[y == cls_index,:]
               samples.append(tmp.mean(dim=0))
               labels.append(cls_index)

           ## region classification
           features = torch.stack(samples, dim=0).unsqueeze(2).unsqueeze(3)
           labels_img = torch.LongTensor(labels).cuda()
           logits_img = self.region_cls_seg(features)
           logits_img = logits_img.squeeze(3).squeeze(2)
           return final_output, seg_label, logits_img, labels_img, region_bias 
        else:
           return final_output + res_output

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
        losses = self.losses(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
        return losses

    def region_cls_seg(self, feat):
        feat = self.region_seg_bn(feat)
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.region_conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, logits_img, labels_img, region_bias):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit_upsample = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        all_logit_upsample = seg_logit_upsample + (region_bias[:,:,:-1]).mean(1).unsqueeze(2).unsqueeze(3)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        # pixel loss
        loss['acc_seg_seg'] = accuracy(seg_logit_upsample, seg_label)
        loss['acc_seg_all'] = accuracy(all_logit_upsample, seg_label)
        loss['loss_seg_seg'] = self.main_w * F.cross_entropy(seg_logit_upsample, seg_label, weight=seg_weight, ignore_index=self.ignore_index)
        loss['loss_seg_all'] = self.main_w * F.cross_entropy(all_logit_upsample, seg_label, weight=seg_weight, ignore_index=self.ignore_index)

        # img aware loss 
        mask = []
        seg_label_t = torch.where(seg_label==self.ignore_index, self.num_classes, seg_label)
        n, h, w = seg_label_t.shape
        region_p_gts = torch.ones(n, self.num_classes + 1, dtype=torch.long).cuda() * self.num_classes
        for i in range(n):
            _classes = torch.unique(seg_label_t[i])
            region_p_gts[i,_classes] = _classes
            img_onehot = F.one_hot(_classes, num_classes=self.num_classes+1).sum(dim=0)
            mask.append(img_onehot)
        region_p_gts = (region_p_gts[:,:-1]).contiguous()
        mask = torch.stack(mask, dim=0)[:,:-1].contiguous().view(n, self.num_classes, 1, 1)
        loss['acc_seg_seg_mask'] = accuracy(seg_logit_upsample * mask, seg_label)
        loss['acc_seg_all_mask'] = accuracy(all_logit_upsample * mask, seg_label)
        loss['loss_seg_all_mask'] = self.main_w * F.cross_entropy(all_logit_upsample * mask, seg_label, weight=seg_weight, ignore_index=self.ignore_index)

        # region_p cls
        mask = ((region_p_gts == self.num_classes) * self.neg_w + region_p_gts != self.num_classes).view(-1,1)
        region_bias = region_bias.view(n * self.num_classes, -1)
        region_p_onehot = F.one_hot(region_p_gts.view(-1,), num_classes=self.num_classes + 1)
        region_p_onehot = self.smooth * region_p_onehot + (1 - self.smooth) / self.num_classes * (1 - region_p_onehot)
        if self.prior:
           region_bias = region_bias + torch.log(self.weight + 1e-12)
        loss['loss_region_p_cls'] = -self.region_w * (mask * region_p_onehot * F.log_softmax(region_bias, dim=1)).sum() / (mask.sum() + 1e-12)

        # region_gt cls 
        region_gts_t = torch.where(labels_img >= self.num_classes, self.num_classes, labels_img)
        mask = ((region_gts_t == self.num_classes) * self.neg_w + region_gts_t != self.num_classes).view(-1,1)
        region_onehot = F.one_hot(region_gts_t, num_classes=self.num_classes + 1)
        region_onehot = self.smooth * region_onehot + (1 - self.smooth) / self.num_classes * (1 - region_onehot)
        if self.prior:
           logits_img = logits_img + torch.log(self.weight + 1e-12)
        loss['loss_img_cls'] = -self.region_w * (mask * region_onehot * F.log_softmax(logits_img, dim=1)).sum() / (mask.sum() + 1e-12)
        return loss


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@HEADS.register_module()
class DepthwiseSeparableASPPHead_regionrebalance(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.
    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.
    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPHead_regionrebalance, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        ## region cls seg
        self.region_seg_bn = nn.Sequential(
             nn.Conv2d(self.channels, self.channels, 1),
             nn.ReLU(inplace=True))
        self.region_conv_seg = nn.Conv2d(self.channels, self.num_classes+1, kernel_size=1)

        ## rebalance
        frequency_file = kwargs.get('frequency_file')
        self.region_w = float(kwargs.get('region_w'))
        self.smooth = float(kwargs.get('smooth'))
        self.main_w = float(kwargs.get('main_w'))
        self.neg_w = float(kwargs.get('neg_w'))
        self.prior = bool(kwargs.get('prior'))

        img_w_list = []
        content = open(frequency_file, "r").readlines()
        for line in content:
            img_w_list.append(int(line))

        weight = torch.Tensor(img_w_list)
        weight = (weight / weight.sum()).view(1, self.num_classes)
        self.weight = torch.ones(1, self.num_classes+1)
        self.weight[:,:-1] = weight
        self.weight = nn.parameter.Parameter(self.weight, requires_grad=False)


    def forward(self, inputs, seg_label=None):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        final_output = self.cls_seg(output)

        # region
        #final_output_t = final_output.permute(0,2,3,1)
        #score = F.gumbel_softmax(final_output_t, tau=1, hard=False).permute(0,3,1,2).flatten(2)
        score = F.softmax(final_output, dim=1).flatten(2)                  # n 150, hw 
        feats = output.flatten(2)                                          # n 512, hw
        res_w = (score @ feats.transpose(1,2)) / (score.sum(dim=2, keepdim=True) + 1e-12)  # n 150 hw, n hw 512 -> n 150 512
        n = output.size(0)
        region_bias = self.region_cls_seg(res_w.view(n * self.num_classes, self.channels, 1, 1))
        region_bias = region_bias.view(n, self.num_classes, self.num_classes + 1)
        res_output = (region_bias[:,:,:-1]).mean(1).view(n, self.num_classes, 1, 1)

        # decoupling
        if self.training:
           n, c, h, w = output.shape
           reduced_seg_label = F.interpolate(seg_label.to(torch.float32), size=(h, w), mode='nearest')
           reduced_seg_label = reduced_seg_label.squeeze(1).long()

           samples = []
           labels = []
           pixel_features = output.permute(0,2,3,1)
           y = reduced_seg_label.squeeze(1)
           _classes = torch.unique(y)
           for cls_index in _classes:
               tmp = pixel_features[y == cls_index,:]
               samples.append(tmp.mean(dim=0))
               labels.append(cls_index)

           ## region classification
           features = torch.stack(samples, dim=0).unsqueeze(2).unsqueeze(3)
           labels_img = torch.LongTensor(labels).cuda()
           logits_img = self.region_cls_seg(features)
           logits_img = logits_img.squeeze(3).squeeze(2)
           return final_output, seg_label, logits_img, labels_img, region_bias 
        else:
           return final_output + res_output

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
        losses = self.losses(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
        return losses

    def region_cls_seg(self, feat):
        feat = self.region_seg_bn(feat)
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.region_conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, logits_img, labels_img, region_bias):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit_upsample = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        all_logit_upsample = seg_logit_upsample + (region_bias[:,:,:-1]).mean(1).unsqueeze(2).unsqueeze(3)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        # pixel loss
        loss['acc_seg_seg'] = accuracy(seg_logit_upsample, seg_label)
        loss['acc_seg_all'] = accuracy(all_logit_upsample, seg_label)
        loss['loss_seg_seg'] = self.main_w * F.cross_entropy(seg_logit_upsample, seg_label, weight=seg_weight, ignore_index=self.ignore_index)
        loss['loss_seg_all'] = self.main_w * F.cross_entropy(all_logit_upsample, seg_label, weight=seg_weight, ignore_index=self.ignore_index)

        # img aware loss 
        mask = []
        seg_label_t = torch.where(seg_label==self.ignore_index, self.num_classes, seg_label)
        n, h, w = seg_label_t.shape
        region_p_gts = torch.ones(n, self.num_classes + 1, dtype=torch.long).cuda() * self.num_classes
        for i in range(n):
            _classes = torch.unique(seg_label_t[i])
            region_p_gts[i,_classes] = _classes
            img_onehot = F.one_hot(_classes, num_classes=self.num_classes+1).sum(dim=0)
            mask.append(img_onehot)
        region_p_gts = (region_p_gts[:,:-1]).contiguous()
        mask = torch.stack(mask, dim=0)[:,:-1].contiguous().view(n, self.num_classes, 1, 1)
        loss['acc_seg_seg_mask'] = accuracy(seg_logit_upsample * mask, seg_label)
        loss['acc_seg_all_mask'] = accuracy(all_logit_upsample * mask, seg_label)
        loss['loss_seg_all_mask'] = self.main_w * F.cross_entropy(all_logit_upsample * mask, seg_label, weight=seg_weight, ignore_index=self.ignore_index)

        # region_p cls
        mask = ((region_p_gts == self.num_classes) * self.neg_w + region_p_gts != self.num_classes).view(-1,1)
        region_bias = region_bias.view(n * self.num_classes, -1)
        region_p_onehot = F.one_hot(region_p_gts.view(-1,), num_classes=self.num_classes + 1)
        region_p_onehot = self.smooth * region_p_onehot + (1 - self.smooth) / self.num_classes * (1 - region_p_onehot)
        if self.prior:
           region_bias = region_bias + torch.log(self.weight + 1e-12)
        loss['loss_region_p_cls'] = -self.region_w * (mask * region_p_onehot * F.log_softmax(region_bias, dim=1)).sum() / (mask.sum() + 1e-12)

        # region_gt cls 
        region_gts_t = torch.where(labels_img >= self.num_classes, self.num_classes, labels_img)
        mask = ((region_gts_t == self.num_classes) * self.neg_w + region_gts_t != self.num_classes).view(-1,1)
        region_onehot = F.one_hot(region_gts_t, num_classes=self.num_classes + 1)
        region_onehot = self.smooth * region_onehot + (1 - self.smooth) / self.num_classes * (1 - region_onehot)
        if self.prior:
           logits_img = logits_img + torch.log(self.weight + 1e-12)
        loss['loss_img_cls'] = -self.region_w * (mask * region_onehot * F.log_softmax(logits_img, dim=1)).sum() / (mask.sum() + 1e-12)
        return loss



class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


@HEADS.register_module()
class OCRHead_regionrebalance(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.
    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.
    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, scale=1, **kwargs):
        super(OCRHead_regionrebalance, self).__init__(**kwargs)
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        ## region cls seg
        self.region_seg_bn = nn.Sequential(
             nn.Conv2d(self.channels, self.channels, 1),
             nn.ReLU(inplace=True))
        self.region_conv_seg = nn.Conv2d(self.channels, self.num_classes+1, kernel_size=1)

        ## rebalance
        frequency_file = kwargs.get('frequency_file')
        self.region_w = float(kwargs.get('region_w'))
        self.smooth = float(kwargs.get('smooth'))
        self.main_w = float(kwargs.get('main_w'))
        self.neg_w = float(kwargs.get('neg_w'))
        self.prior = bool(kwargs.get('prior'))

        img_w_list = []
        content = open(frequency_file, "r").readlines()
        for line in content:
            img_w_list.append(int(line))

        weight = torch.Tensor(img_w_list)
        weight = (weight / weight.sum()).view(1, self.num_classes)
        self.weight = torch.ones(1, self.num_classes+1)
        self.weight[:,:-1] = weight
        self.weight = nn.parameter.Parameter(self.weight, requires_grad=False)


    def forward(self, inputs, prev_output, seg_label=None):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        output = object_context
        final_output = self.cls_seg(output)


        # region
        #final_output_t = final_output.permute(0,2,3,1)
        #score = F.gumbel_softmax(final_output_t, tau=1, hard=False).permute(0,3,1,2).flatten(2)
        score = F.softmax(final_output, dim=1).flatten(2)                  # n 150, hw 
        feats = output.flatten(2)                                          # n 512, hw
        res_w = (score @ feats.transpose(1,2)) / (score.sum(dim=2, keepdim=True) + 1e-12)  # n 150 hw, n hw 512 -> n 150 512
        n = output.size(0)
        region_bias = self.region_cls_seg(res_w.view(n * self.num_classes, self.channels, 1, 1))
        region_bias = region_bias.view(n, self.num_classes, self.num_classes + 1)
        res_output = (region_bias[:,:,:-1]).mean(1).view(n, self.num_classes, 1, 1)

        # decoupling
        if self.training:
           n, c, h, w = output.shape
           reduced_seg_label = F.interpolate(seg_label.to(torch.float32), size=(h, w), mode='nearest')
           reduced_seg_label = reduced_seg_label.squeeze(1).long()

           samples = []
           labels = []
           pixel_features = output.permute(0,2,3,1)
           y = reduced_seg_label.squeeze(1)
           _classes = torch.unique(y)
           for cls_index in _classes:
               tmp = pixel_features[y == cls_index,:]
               samples.append(tmp.mean(dim=0))
               labels.append(cls_index)

           ## region classification
           features = torch.stack(samples, dim=0).unsqueeze(2).unsqueeze(3)
           labels_img = torch.LongTensor(labels).cuda()
           logits_img = self.region_cls_seg(features)
           logits_img = logits_img.squeeze(3).squeeze(2)
           return final_output, seg_label, logits_img, labels_img, region_bias 
        else:
           return final_output + res_output


    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg,
                      train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
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
        outputs = self.forward(inputs, prev_output, gt_semantic_seg)
        losses = self.losses(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
        return losses

    def region_cls_seg(self, feat):
        feat = self.region_seg_bn(feat)
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.region_conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, logits_img, labels_img, region_bias):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit_upsample = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        all_logit_upsample = seg_logit_upsample + (region_bias[:,:,:-1]).mean(1).unsqueeze(2).unsqueeze(3)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        # pixel loss
        loss['acc_seg_seg'] = accuracy(seg_logit_upsample, seg_label)
        loss['acc_seg_all'] = accuracy(all_logit_upsample, seg_label)
        loss['loss_seg_seg'] = self.main_w * F.cross_entropy(seg_logit_upsample, seg_label, weight=seg_weight, ignore_index=self.ignore_index)
        loss['loss_seg_all'] = self.main_w * F.cross_entropy(all_logit_upsample, seg_label, weight=seg_weight, ignore_index=self.ignore_index)

        # img aware loss 
        mask = []
        seg_label_t = torch.where(seg_label==self.ignore_index, self.num_classes, seg_label)
        n, h, w = seg_label_t.shape
        region_p_gts = torch.ones(n, self.num_classes + 1, dtype=torch.long).cuda() * self.num_classes
        for i in range(n):
            _classes = torch.unique(seg_label_t[i])
            region_p_gts[i,_classes] = _classes
            img_onehot = F.one_hot(_classes, num_classes=self.num_classes+1).sum(dim=0)
            mask.append(img_onehot)
        region_p_gts = (region_p_gts[:,:-1]).contiguous()
        mask = torch.stack(mask, dim=0)[:,:-1].contiguous().view(n, self.num_classes, 1, 1)
        loss['acc_seg_seg_mask'] = accuracy(seg_logit_upsample * mask, seg_label)
        loss['acc_seg_all_mask'] = accuracy(all_logit_upsample * mask, seg_label)
        loss['loss_seg_all_mask'] = self.main_w * F.cross_entropy(all_logit_upsample * mask, seg_label, weight=seg_weight, ignore_index=self.ignore_index)

        # region_p cls
        mask = ((region_p_gts == self.num_classes) * self.neg_w + region_p_gts != self.num_classes).view(-1,1)
        region_bias = region_bias.view(n * self.num_classes, -1)
        region_p_onehot = F.one_hot(region_p_gts.view(-1,), num_classes=self.num_classes + 1)
        region_p_onehot = self.smooth * region_p_onehot + (1 - self.smooth) / self.num_classes * (1 - region_p_onehot)
        if self.prior:
           region_bias = region_bias + torch.log(self.weight + 1e-12)
        loss['loss_region_p_cls'] = -self.region_w * (mask * region_p_onehot * F.log_softmax(region_bias, dim=1)).sum() / (mask.sum() + 1e-12)

        # region_gt cls 
        region_gts_t = torch.where(labels_img >= self.num_classes, self.num_classes, labels_img)
        mask = ((region_gts_t == self.num_classes) * self.neg_w + region_gts_t != self.num_classes).view(-1,1)
        region_onehot = F.one_hot(region_gts_t, num_classes=self.num_classes + 1)
        region_onehot = self.smooth * region_onehot + (1 - self.smooth) / self.num_classes * (1 - region_onehot)
        if self.prior:
           logits_img = logits_img + torch.log(self.weight + 1e-12)
        loss['loss_img_cls'] = -self.region_w * (mask * region_onehot * F.log_softmax(logits_img, dim=1)).sum() / (mask.sum() + 1e-12)
        return loss
