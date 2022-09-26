import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy

class PaCoLoss(nn.Module):
    def __init__(self, alpha=0.05, beta=1.0, gamma=1.0, supt=1.0, temperature=0.2, base_temperature=None, K=8192, num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 
        self.supt = supt
        self.num_classes = num_classes

    def forward(self, features, labels=None, sup_logits=None, smooth=0.1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        one_hot_label = smooth / (self.num_classes - 1 ) * (1 - one_hot_label) + (1 - smooth) * one_hot_label
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000, smooth=0.1):
        super(MultiTaskLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.effective_num_beta = 0.999
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smooth)


    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        loss_ce = self.criterion(sup_logits, labels[:batch_size].squeeze())
        return loss_ce + self.alpha * loss
