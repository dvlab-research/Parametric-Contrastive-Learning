# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule

from .paco import PaCoLoss


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
class DepthwiseSeparableASPPHead(ASPPHead):
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
        super(DepthwiseSeparableASPPHead, self).__init__(**kwargs)
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

        self.is_paco = bool(kwargs.get('is_paco', False))
        if self.is_paco:
            # paco loss
            self.mlp = nn.Sequential(
               ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,inplace=False),
               ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,inplace=False),
               nn.Conv2d(self.channels, 128, 1))
            self.alpha = float(kwargs.get('alpha'))
            self.temperature = float(kwargs.get('temperature'))
            self.K = int(kwargs.get('K'))
            self.paco_loss = PaCoLoss(alpha=self.alpha, num_classes=self.num_classes, temperature=self.temperature)


    def forward(self, inputs):
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
        seg_logit = self.cls_seg(output)

        if self.training and self.is_paco:
            embed = self.mlp(output)
            return seg_logit, embed 
        else:
            return seg_logit


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        if self.is_paco:
            seg_logit, embed = self.forward(inputs)
        else:
            seg_logit = self.forward(inputs)
        losses = self.losses(seg_logit, gt_semantic_seg)

        if self.is_paco:
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

