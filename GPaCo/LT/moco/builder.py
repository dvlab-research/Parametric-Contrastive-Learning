"""https://github.com/facebookresearch/moco"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class NormedLinear_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048):
        super(NormedLinear_Classifier, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, *args):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


def flatten(t):
    return t.reshape(t.shape[0], -1)

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.2, mlp=False, feat_dim=2048, num_classes=1000):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.linear = nn.Linear(feat_dim, num_classes)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_q.fc)


        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # cross_entropy
        self.layer = -2 
        self.feat_after_avg_q = None
        self._register_hook()
        self.normalize = False 

    
    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]

            return children[self.layer]
        return None

    def _hook_q(self, _, __, output):
        self.feat_after_avg_q = flatten(output)
        if self.normalize:
           self.feat_after_avg_q = nn.functional.normalize(self.feat_after_avg_q, dim=1)

    def _register_hook(self):
        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_q.register_forward_hook(self._hook_q)

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


    def _train(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        logits_q = self.linear(self.feat_after_avg_q)

        # compute key features
        k = self.encoder_q(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)
        logits_k = self.linear(self.feat_after_avg_q)

        # compute logits
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, labels)

        # compute logits 
        logits = torch.cat((logits_q, logits_k), dim=0)

        return features, target, logits

    def _inference(self, image):
        q = self.encoder_q(image)
        encoder_q_logits = self.linear(self.feat_after_avg_q)
        return encoder_q_logits

    def forward(self, im_q, im_k=None, labels=None):
        if self.training:
           return self._train(im_q, im_k, labels) 
        else:
           return self._inference(im_q)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
