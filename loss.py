"""Loss functions for binary image segmentation.

This module provides:
- **MCCLoss** — Matthews Correlation Coefficient loss (Abhishek & Hamarneh,
  IEEE ISBI 2021).  Unlike Dice / IoU losses, MCC uses all four entries of
  the confusion matrix (TP, TN, FP, FN), making it effective when the
  background class dominates the image.
- **Dice_Loss** — Sørensen–Dice coefficient loss.

Paper : https://doi.org/10.1109/ISBI48211.2021.9433782
GitHub: https://github.com/kakumarabhishek/MCC-Loss
"""

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["MCCLoss", "Dice_Loss"]


# ---------------------------------------------------------------------------
# MCC Loss
# ---------------------------------------------------------------------------


class MCCLoss(nn.Module):
    """
    Matthews Correlation Coefficient loss for binary segmentation.

    Computes ``L_MCC = 1 - MCC`` where::

        MCC = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    The soft confusion-matrix entries are computed as in the paper (Eq. 5)::

        TP = sum(y_pred * y_true)
        TN = sum((1 - y_pred) * (1 - y_true))
        FP = sum(y_pred * (1 - y_true))
        FN = sum((1 - y_pred) * y_true)

    Args:
        from_logits: If ``True``, apply sigmoid to ``y_pred`` before
            computing the loss.  Default: ``False``.
        reduction: ``"batch"`` -- flatten the whole batch into one MCC
            computation; ``"sample"`` -- compute per-sample MCC and average.
            Default: ``"batch"``.
        eps: Small constant for numerical stability inside the square root.
            Default: ``1e-7``.

    Shapes:
        - y_pred: ``(N, 1, ...)`` or ``(N, ...)`` -- predicted probabilities
          in [0, 1] (or logits if ``from_logits=True``).
        - y_true: same shape as *y_pred* -- binary ground-truth (0 or 1).
        - Output: scalar loss.

    Example::

        >>> criterion = MCCLoss()
        >>> y_pred = torch.rand(4, 1, 128, 128)
        >>> y_true = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> loss = criterion(y_pred, y_true)

        # With raw logits (before sigmoid):
        >>> criterion = MCCLoss(from_logits=True)
        >>> logits = model(images)          # (N, 1, H, W)
        >>> loss = criterion(logits, masks)

        # Simple / legacy-style usage (flat tensors, same as the original
        # MCC_Loss class):
        >>> criterion = MCCLoss()           # defaults: from_logits=False, reduction="batch"
        >>> pred_mask = torch.tensor([[0., 1., 1., 0., 0.]])
        >>> gt_mask   = torch.tensor([[0., 0., 1., 0., 0.]])
        >>> loss = criterion(pred_mask, gt_mask)

    Reference:
        K. Abhishek and G. Hamarneh, "Matthews Correlation Coefficient Loss
        for Deep Convolutional Networks: Application to Skin Lesion
        Segmentation," IEEE ISBI, 2021, pp. 225-229.
    """

    def __init__(
        self,
        from_logits: bool = False,
        reduction: str = "batch",
        eps: float = 1e-7,
    ):
        super().__init__()
        if reduction not in ("sample", "batch"):
            raise ValueError(
                f"reduction must be 'sample' or 'batch', got '{reduction}'"
            )
        self.from_logits = from_logits
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the MCC loss.

        Args:
            y_pred: Predicted probabilities or logits.
                Shape: ``(N, 1, ...)`` or ``(N, ...)``.
            y_true: Binary ground-truth labels (0 or 1).
                Same shape as *y_pred*.

        Returns:
            Scalar loss value.

        Raises:
            ValueError: On shape mismatch or fewer than 2 dimensions.
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
            )
        if y_pred.ndim < 2:
            raise ValueError(
                f"Expected at least 2D input (N, ...), got {y_pred.ndim}D"
            )

        y_pred = y_pred.float()
        y_true = y_true.float()

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        # Flatten everything after the batch dimension.
        bs = y_pred.shape[0]
        y_pred = y_pred.view(bs, -1)
        y_true = y_true.view(bs, -1)

        if self.reduction == "batch":
            y_pred = y_pred.view(-1)
            y_true = y_true.view(-1)
            dim = 0
        else:  # "sample"
            dim = 1

        # Soft confusion-matrix entries (Eq. 5 in the paper).
        tp = (y_pred * y_true).sum(dim=dim)
        tn = ((1 - y_pred) * (1 - y_true)).sum(dim=dim)
        fp = (y_pred * (1 - y_true)).sum(dim=dim)
        fn = ((1 - y_pred) * y_true).sum(dim=dim)

        # MCC (Eq. 3) and loss (Eq. 4).
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + self.eps
        )

        mcc = numerator / denominator
        loss = 1.0 - mcc

        # When fp = fn = 0 the prediction is perfect, but the formula gives
        # mcc ≈ 0 (not 1) whenever tp = 0 or tn = 0 because both numerator
        # and the denominator product tend to 0.  Correct those entries.
        perfect = (fp == 0) & (fn == 0)
        loss = torch.where(perfect, torch.zeros_like(loss), loss)

        return loss.mean()


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """
    Sørensen–Dice coefficient-based loss.

    Computes ``L_Dice = 1 - Dice`` where::

        Dice = (2 * TP) / (2 * TP + FP + FN)
    
    Args:
        inputs (Tensor): Predicted probabilities (values in [0, 1]).
        targets (Tensor): Binary ground-truth labels (0 or 1).

    Reference:
        https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/loss.py#L28
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        mul = torch.mul(inputs, targets)
        add = torch.add(inputs, targets)
        dice = 2 * torch.div(mul.sum(), add.sum())
        return 1 - dice
