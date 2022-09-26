# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer_paco(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, paco=True, **kwargs):
        super(VisionTransformer_paco, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.paco = paco
        if self.paco:
            embed_dim = kwargs['embed_dim']
            self.K = 8192
            self.mlp = nn.Sequential(
                 nn.Linear(embed_dim, embed_dim),
                 nn.GELU(),
                 nn.Linear(embed_dim, embed_dim),
                 nn.GELU(),
                 nn.Linear(embed_dim, 128))

            self.register_buffer("queue", torch.randn(self.K, 128))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_l", torch.randint(0, 1000, (self.K,)))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size,:] = keys
        self.queue_l[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, x, y=None):
        if isinstance(x, list):
            x_q, x_k = x[0], x[1]
            x_q, x_k = self.forward_features(x_q), self.forward_features(x_k)
            dual_view = True
        else:
            x_q = self.forward_features(x)
            dual_view = False

        if self.paco and self.training:
           if not dual_view:
              q = self.mlp(x_q)
              q = nn.functional.normalize(q, dim=1)
              features = torch.cat((q, self.queue.clone().detach()), dim=0)
              target = torch.cat((y, self.queue_l.clone().detach()), dim=0)
              self._dequeue_and_enqueue(q, y)
              x = self.head(x_q)
           else:
              q, k = self.mlp(x_q), self.mlp(x_k)
              q, k = nn.functional.normalize(q, dim=1), nn.functional.normalize(k, dim=1)
              features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
              target = torch.cat((y, y, self.queue_l.clone().detach()), dim=0)
              self._dequeue_and_enqueue(k, y)
              x1, x2 = self.head(x_q), self.head(x_k)
              x = torch.cat((x1,x2), dim=0)

        if self.training:
           return features, target, x
        else:
            x = self.head(x_q)
            return x


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer_paco(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer_paco(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer_paco(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
