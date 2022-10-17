import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, weight=None, size_average=True, num_classes=150, ignore_index=255):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.num_classes=num_classes
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = (target != self.ignore_index)
        gt = torch.where(mask, target, self.num_classes)
        one_hot_gt = F.one_hot(gt, num_classes=self.num_classes+1)[:,:,:,:-1].permute(0,3,1,2).contiguous()
        score = F.softmax(input, dim=1)

        # focal loss
        score_gt = (one_hot_gt * score).sum(1)
        loss = -((1-score_gt) ** self.gamma) * (one_hot_gt * torch.log(score + 1e-12)).sum(1)
        loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-12)
        return loss
