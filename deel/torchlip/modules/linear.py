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
import torch
from torch.nn.utils.parametrizations import spectral_norm

from ..utils import bjorck_norm
from ..normalizers import DEFAULT_EPS_BJORCK
from ..normalizers import DEFAULT_EPS_SPECTRAL
from ..utils import frobenius_norm
from .module import LipschitzModule


class SpectralLinear(torch.nn.Linear, LipschitzModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        k_coef_lip: float = 1.0,
        eps_spectral: int = DEFAULT_EPS_SPECTRAL,
        eps_bjorck: int = DEFAULT_EPS_BJORCK,
    ):
        """
        This class is a Linear Layer constrained such that all singular of it's kernel
        are 1. The computation based on BjorckNormalizer algorithm.
        The computation is done in two steps:

        1. reduce the larget singular value to 1, using iterated power method.
        2. increase other singular values to 1, using BjorckNormalizer algorithm.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If ``False``, the layer will not learn an additive bias.
            k_coef_lip: Lipschitz constant to ensure.
            eps_spectral: stopping criterion for the iterative power algorithm.
            eps_bjorck: stopping criterion Bjorck algorithm.

        Shape:
            - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
              additional dimensions and :math:`H_{in} = \\text{in\\_features}`
            - Output: :math:`(N, *, H_{out})` where all but the last dimension
              are the same shape as the input and
              :math:`H_{out} = \\text{out\\_features}`.

        This documentation reuse the body of the original torch.nn.Linear doc.
        """
        torch.nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        LipschitzModule.__init__(self, k_coef_lip)

        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

        spectral_norm(
            self,
            name="weight",
            eps=eps_spectral,
        )
        bjorck_norm(self, name="weight", eps=eps_bjorck)
        self.apply_lipschitz_factor()

    def vanilla_export(self) -> torch.nn.Linear:
        layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
        )
        layer.weight.data = self.weight.detach()
        if self.bias is not None:
            layer.bias.data = self.bias.detach()
        return layer


class FrobeniusLinear(torch.nn.Linear, LipschitzModule):
    """
    This class is a Linear Layer constrained such that the Frobenius norm of the weight
    is 1. In the case of a single output neuron, it is equivalent and faster than the
    SpectralLinear layer. For multi-neuron case, the "disjoint_neurons" parameter
    affects the behaviour:

    - if ``disjoint_neurons`` is True (default), it corresponds to the stacking of
      independent 1-Lipschitz neurons.
    - if ``disjoint_neurons`` is False, the matrix weight is normalized by its Frobenius
      norm.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``False``, the layer will not learn an additive bias.
        disjoint_neurons: Normalize, independently per neuron or not, the matrix weight.
        k_coef_lip: Lipschitz constant to ensure.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        disjoint_neurons: bool = True,
        k_coef_lip: float = 1.0,
    ):
        torch.nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        LipschitzModule.__init__(self, k_coef_lip)

        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

        frobenius_norm(self, name="weight", disjoint_neurons=disjoint_neurons)
        self.apply_lipschitz_factor()

    def vanilla_export(self):
        layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
        )
        layer.weight.data = self.weight.detach()
        if self.bias is not None:
            layer.bias.data = self.bias.detach()
        return layer
