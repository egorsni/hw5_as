import torch
from torch import Tensor
from torch.nn import CTCLoss
from torch.nn import CrossEntropyLoss


class CELoss(CTCLoss):
    def forward(self, pred, real, **batch) -> Tensor:
        return CrossEntropyLoss(weight=torch.tensor([1.,9.]).to('cuda'))(pred, real)