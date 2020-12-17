# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains extra activation functions which respect the Lipschitz constant.
It can be added as a layer, or it can be used in the "activation" params for other
layers.
"""

from typing import Optional

import torch
import torch.nn as nn

from .layers import LipschitzModule
from . import functional as F


class MaxMin(nn.Module, LipschitzModule):
    def __init__(self, k_coef_lip: float = 1.0):
        """
        MaxMin activation [ReLU(x),ReLU(-x)]

        Args:
            k_coef_lip: The lipschitz coefficient to enforce.

        Input shape:
            Arbitrary.

        Output shape:
            Double channel size as input.

        References:
            ([M. Blot, M. Cord, et N. Thome, « Max-min convolutional neural networks
            for image classification », in 2016 IEEE International Conference on Image
            Processing (ICIP), Phoenix, AZ, USA, 2016, p. 3678‑3682.)

        """
        nn.Module.__init__(self)
        LipschitzModule.__init__(self, k_coef_lip, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.max_min(input, self._coefficient)

    def vanilla_export(self):
        return self


class GroupSort(nn.Module, LipschitzModule):
    def __init__(self, group_size: Optional[int] = None, k_coef_lip: float = 1.0):
        """
        GroupSort activation

        Args:
            n: group size used when sorting. When None group size is set to input
                size (fullSort behavior)
            data_format: either channels_first or channels_last
            k_coef_lip: the lipschitz coefficient to be enforced
            *args: params passed to Layers
            **kwargs: params passed to layers (named fashion)

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        nn.Module.__init__(self)
        LipschitzModule.__init__(self, k_coef_lip, 1.0)
        self.group_size = group_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.group_sort(input, self.group_size, self._coefficient)

    def vanilla_export(self):
        return self


class GroupSort2(GroupSort):
    def __init__(self, k_coef_lip: float = 1.0):
        """
        GroupSort2 activation. Special case of GroupSort with group of size 2.

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        super().__init__(group_size=2, k_coef_lip=k_coef_lip)


class FullSort(GroupSort):
    def __init__(self, k_coef_lip: float = 1.0):
        """
        FullSort activation. Special case of GroupSort where the entire input is sorted.

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        super().__init__(group_size=None, k_coef_lip=k_coef_lip)


class LipschitzPReLU(nn.PReLU, LipschitzModule):
    """
    PreLu activation, with Lipschitz constraint.

    Args:
        k_coef_lip: lipschitz coefficient to be enforced
    """

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, k_coef_lip: float = 1.0
    ):
        nn.PReLU.__init__(self, num_parameters=num_parameters, init=init)
        LipschitzModule.__init__(self, k_coef_lip, 1.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.lipschitz_prelu(input, self.weight, self._coefficient)

    def vanilla_export(self):
        layer = LipschitzPReLU(num_parameters=self.num_parameters)
        layer.weight.data = self.weight.data
        return layer
