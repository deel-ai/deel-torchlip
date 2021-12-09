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
from torch.nn.utils import spectral_norm

from ..utils import bjorck_norm
from ..utils import DEFAULT_NITER_BJORCK
from ..utils import DEFAULT_NITER_SPECTRAL
from ..utils import frobenius_norm
from .module import LipschitzModule


class SpectralLinear(torch.nn.Linear, LipschitzModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        k_coef_lip: float = 1.0,
        niter_spectral: int = DEFAULT_NITER_SPECTRAL,
        niter_bjorck: int = DEFAULT_NITER_BJORCK,
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
            niter_spectral: Number of iteration to find the maximum singular value.
            niter_bjorck: Number of iteration with BjorckNormalizer algorithm.

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
            n_power_iterations=niter_spectral,
        )
        bjorck_norm(self, name="weight", n_iterations=niter_bjorck)
        self.register_forward_pre_hook(self._hook)

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
    Same a SpectralLinear, but in the case of a single output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
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

        frobenius_norm(self, name="weight")
        self.register_forward_pre_hook(self._hook)

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
