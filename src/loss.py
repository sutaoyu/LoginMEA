import torch
from torch import nn

try:
    from models import *
    from utils import *
except:
    from src.models import *
    from src.utils import *


class CustomMultiLossLayer(nn.Module):
    def __init__(self, loss_num, device=None):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(
            torch.zeros(
                self.loss_num,
            ),
            requires_grad=True,
        )

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss


class InfoNCE_loss(nn.Module):
    def __init__(self, device, temperature=0.05) -> None:
        super().__init__()
        self.device = device
        self.t = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def sim(self, emb_left, emb_right):
        return emb_left.mm(emb_right.t())

    def forward(self, emb, train_links):
        emb = F.normalize(emb)
        emb_train_left = emb[train_links[:, 0]]
        emb_train_right = emb[train_links[:, 1]]
        score = self.sim(emb_train_left, emb_train_right)
        bsize = emb_train_left.size()[0]
        label = torch.arange(bsize, dtype=torch.long).cuda(self.device)
        loss = self.ce_loss(score / self.t, label)
        return loss
