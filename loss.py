import torch
import torch.nn as nn
from torch.nn import functional as F


class Dice_Loss(nn.Module):
    """
    Calculates the Sørensen–Dice coefficient-based loss.
    Taken from 
    https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/loss.py#L28

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """
    def __init__(self):
        super(Dice_Loss, self).__init__()

    def forward(self, inputs, targets):
        mul = torch.mul(inputs, targets)
        add = torch.add(inputs, 1, targets)
        dice = 2 * torch.div(mul.sum(), add.sum())
        return 1 - dice

class MCC_Loss(nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """
    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))
        
        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp) * torch.add(tp, 1, fn) * torch.add(tn, 1, fp) * torch.add(tn, 1, fn)
        )
        
        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.)
        return 1 - mcc