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
"""
This module contains extra activation functions which respect the Lipschitz constant.
It can be added as a layer, or it can be used in the "activation" params for other
layers.
"""
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from .. import functional as F
from .module import LipschitzModule


class MaxMin(nn.Module, LipschitzModule):
    r"""
    Applies max-min activation.

    If ``input`` is a tensor of shape :math:`(N, C)` and ``dim`` is
    ``None``, the output can be described as:

    .. math::
        \text{out}(N_i, C_{2j}) = \max(\text{input}(N_i, C_j), 0)\\
        \text{out}(N_i, C_{2j + 1}) = \max(-\text{input}(N_i, C_j), 0)

    where :math:`N` is the batch size and :math:`C` is the size of the
    tensor.

    See also :func:`.functional.max_min`.
    """

    def __init__(self, dim: Optional[int] = None, k_coef_lip: float = 1.0):
        r"""
        Args:
            dim: The dimension to apply max-min. If None, will apply to the
                0th dimension if the shape of input is :math:`(C)` or to the
                first if its :math:`(N, C, *)`.
            k_coef_lip: The lipschitz coefficient to enforce.

        Shape:
            - Input: :math:`(C)` or :math:`(N, C, *)` where :math:`*` means
              any number of additional dimensions.
            - Output: :math:`(2C)` is the input shape was :math:`(C)`, or
              :math:`(N, 2C, *)` if ``dim`` is ``None``, otherwise
              :math:`(N, *, 2C_{dim}, *)` where :math:`C_{dim}` is the
              dimension corresponding to the ``dim`` parameter.

        Note:
            M. Blot, M. Cord, et N. Thome, « Max-min convolutional neural networks
            for image classification », in 2016 IEEE International Conference on Image
            Processing (ICIP), Phoenix, AZ, USA, 2016, p. 3678‑3682.

        """
        nn.Module.__init__(self)
        LipschitzModule.__init__(self, k_coef_lip)
        self._dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.max_min(input, self._dim) * self._coefficient_lip

    def vanilla_export(self):
        return self


class GroupSort(nn.Module, LipschitzModule):
    r"""
    Applies group-sort activation.

    The activation works by first reshaping the input to a tensor
    of shape :math:`(N', G)` where :math:`G` is the group size and
    :math:`N'` the number of groups, then sorting each group of
    size :math:`G` and then reshaping to the original input shape.

    See also :func:`.functional.group_sort`.
    """

    def __init__(self, group_size: Optional[int] = None, k_coef_lip: float = 1.0):
        """
        Args:
            group_size: group size used when sorting. When None group size
            is set to input size (fullSort behavior)
            data_format: either channels_first or channels_last
            k_coef_lip: The lipschitz coefficient to enforce.

        Shape:
            - Input: :math:`(N,∗)` where :math:`*` means, any number
               of additional dimensions
            - Output: :math:`(N,*)`, same shape as the input.

        Example:
            >>> m = torch.randn(2, 4)
            tensor([[ 0.2805, -2.0528,  0.6478,  0.5745],
                    [-1.4075,  0.0435, -1.2408,  0.2945]])
            >>> torchlip.GroupSort(4)(m)
            tensor([[-2.0528,  0.2805,  0.5745,  0.6478],
                    [-1.4075, -1.2408,  0.0435,  0.2945]])
        """
        nn.Module.__init__(self)
        LipschitzModule.__init__(self, k_coef_lip)
        self.group_size = group_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.group_sort(input, self.group_size) * self._coefficient_lip

    def vanilla_export(self):
        return self


class GroupSort2(GroupSort):
    r"""
    Applies group-sort activation with a group size of 2.

    See :class:`GroupSort` for details.

    See also :func:`.functional.group_sort_2`.
    """

    def __init__(self, k_coef_lip: float = 1.0):
        """
        Args:
            k_coef_lip: The lipschitz coefficient to enforce.
        """
        super().__init__(group_size=2, k_coef_lip=k_coef_lip)


class FullSort(GroupSort):
    r"""
    Applies full-sort activation. This is equivalent to group-sort with
    a group-size equals to the size of the input.

    See :class:`GroupSort` for details.

    See also :func:`.functional.full_sort`.
    """

    def __init__(self, k_coef_lip: float = 1.0):
        """
        Args:
            k_coef_lip: The lipschitz coefficient to enforce.
        """
        super().__init__(group_size=None, k_coef_lip=k_coef_lip)


class LPReLU(nn.PReLU, LipschitzModule):
    r"""
    Applies element-wise PReLU activation with Lipschitz constraint:

    .. math::
        LPReLU(x) = \max(0, x) + a' * \min(0, x)

    or

    .. math::
        LPReLU(x) =
        \text{LipschitzPReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        a' * x, & \text{ otherwise }
        \end{cases}

    where :math:`a' = \max(\min(a, k), -k)`, and :math:`a` is a learnable
    parameter.

    See also :class:`torch.nn.PReLU` and :func:`.functional.lipschitz_prelu`.
    """

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, k_coef_lip: float = 1.0
    ):
        """
        Args:
            num_parameters: Number of :math:`a` to learn. Although it
            takes an ``int`` as input, ` there are only two legitimate
            values: 1, or the number of channels at input.
            init: The initial value of :math:`a`.
            k_coef_lip: The lipschitz coefficient to enforce.
        """
        nn.PReLU.__init__(self, num_parameters=num_parameters, init=init)
        LipschitzModule.__init__(self, k_coef_lip)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.lipschitz_prelu(input, self.weight, self._coefficient_lip)

    def vanilla_export(self):
        layer = LPReLU(num_parameters=self.num_parameters)
        layer.weight.data = self.weight.data
        return layer


class HouseHolder(nn.Module, LipschitzModule):
    def __init__(self, channels, k_coef_lip: float = 1.0, theta_initializer=None):
        """
        Householder activation:
        [this review](https://openreview.net/pdf?id=tD7eCtaSkR)
        Adapted from [this repository](https://github.com/singlasahil14/SOC)
        """
        nn.Module.__init__(self)
        LipschitzModule.__init__(self, k_coef_lip)
        assert (channels % 2) == 0
        eff_channels = channels // 2

        if isinstance(theta_initializer, float):
            coef_theta = theta_initializer
        else:
            coef_theta = 0.5 * np.pi
        self.theta = nn.Parameter(
            coef_theta * torch.ones(eff_channels), requires_grad=True
        )
        if theta_initializer is not None:
            if isinstance(theta_initializer, str):
                name2init = {
                    "zeros": torch.nn.init.zeros_,
                    "ones": torch.nn.init.ones_,
                    "normal": torch.nn.init.normal_,
                }
                assert (
                    theta_initializer in name2init
                ), f"Unknown initializer {theta_initializer}"
                name2init[theta_initializer](self.theta)
            elif isinstance(theta_initializer, float):
                pass
            else:
                raise ValueError(f"Unknown initializer {theta_initializer}")

    def forward(self, z, axis=1):
        theta = self.theta.to(z.device).view(1, -1)
        for _ in range(len(z.shape) - len(theta.shape)):
            theta = theta.unsqueeze(-1)
        x, y = z.split(z.shape[axis] // 2, axis)
        selector = (x * torch.sin(0.5 * theta)) - (y * torch.cos(0.5 * theta))

        a_2 = x * torch.cos(theta) + y * torch.sin(theta)
        b_2 = x * torch.sin(theta) - y * torch.cos(theta)

        a = x * (selector <= 0) + a_2 * (selector > 0)
        b = y * (selector <= 0) + b_2 * (selector > 0)
        return torch.cat([a, b], dim=axis)

    def vanilla_export(self):
        return self
