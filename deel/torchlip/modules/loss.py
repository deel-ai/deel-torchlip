# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from typing import Tuple

import torch

from .. import functional as F


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


class KRMulticlassLoss(torch.nn.Module):
    """
    The Wasserstein multiclass loss between ``input`` and ``target``.
    """

    def __init__(self, true_values: Tuple[int, int] = (0, 1)):
        """
        Args:
            true_values: tuple containing the two label for each predicted class.
        """
        self.true_values = true_values

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.kr_multiclass_loss(input, target)


class HingeMulticlassLoss(torch.nn.Module):
    """
    Loss to estimate the Hinge loss in a multiclass setup. It compute the
    elementwise hinge term. This class use pytorch implementation:
    torch.nn.functional.hinge_embedding_loss
    """

    def __init__(self, min_margin: float = 1.0):
        """
        Args:
            min_margin: The minimal margin to enforce.
        """
        self.min_margin = min_margin

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.hinge_multiclass_loss(input, target, self.min_margin)


class HKRMulticlassLoss(torch.nn.Module):
    """
    Loss that estimate the Wasserstein-1 distance using the Kantorovich-Rubinstein
    duality with a hinge regularization.
    """

    def __init__(
        self,
        alpha: float,
        min_margin: float = 1.0,
    ):
        """
        Args:
            alpha: Regularization factor between the hinge and the KR loss.
            min_margin: Minimal margin for the hinge loss.
            true_values: tuple containing the two label for each predicted class.
        """
        self.alpha = alpha
        self.min_margin = min_margin

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.hkr_multiclass_loss(input, target, self.alpha, self.min_margin)
