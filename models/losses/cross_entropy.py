# encoding:utf-8
from torch import nn
from torch.nn import functional as F
import torch


class CrossEntropyLossWithMask(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithMask, self).__init__()

    def forward(self, output_logits, tags, masks):
        output_logits = F.log_softmax(output_logits, dim=2)
        loss = F.nll_loss(output_logits.view(-1, output_logits.size()
                                             [-1]), tags.view(-1), reduction='none')
        loss = loss.view(masks.size()[0], masks.size()[1])
        loss = torch.mul(loss, masks)
        return torch.mean(loss)
