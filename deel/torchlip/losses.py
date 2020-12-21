# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

from typing import Tuple

import torch

from . import functional as F


class KRLoss(torch.nn.Module):
    """
    Loss that estimate the Wasserstein-1 distance using the Kantorovich-Rubinstein
    duality.
    """

    def __init__(self, true_values: Tuple[int, int] = (0, 1)):
        """
        Args:
            true_values: tuple containing the two label for each predicted class.
        """
        self.true_values = true_values

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.kr_loss(input, target, self.true_values)


class NegKRLoss(torch.nn.Module):
    """
    Loss that estimate the negative of the Wasserstein-1 distance using
    the Kantorovich-Rubinstein duality.
    """

    def __init__(self, true_values: Tuple[int, int] = (0, 1)):
        """
        Args:
            true_values: tuple containing the two label for each predicted class.
        """
        self.true_values = true_values

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.neg_kr_loss(input, target, self.true_values)


class HingeMarginLoss(torch.nn.Module):
    """
    Hinge margin loss.
    """

    def __init__(self, min_margin: float = 1.0):
        """
        Args:
            min_margin: The minimal margin to enforce.
        """
        self.min_margin = min_margin

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.hinge_margin_loss(input, target, self.min_margin)


class HKRLoss(torch.nn.Module):
    """
    Loss that estimate the Wasserstein-1 distance using the Kantorovich-Rubinstein
    duality with a hinge regularization.
    """

    def __init__(
        self,
        alpha: float,
        min_margin: float = 1.0,
        true_values: Tuple[int, int] = (-1, 1),
    ):
        """
        Args:
            alpha: Regularization factor between the hinge and the KR loss.
            min_margin: Minimal margin for the hinge loss.
            true_values: tuple containing the two label for each predicted class.
        """
        self.alpha = alpha
        self.min_margin = min_margin
        self.true_values = true_values

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.hkr_loss(input, target, self.alpha, self.min_margin, self.true_values)
