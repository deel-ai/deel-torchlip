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
Custom autograd function for safe-gradient computation of square-root at 0.
"""
from typing import Any

import torch


class SqrtEpsGrad(torch.autograd.Function):
    """
    Small class to avoid division by zero when computing the gradient
    of the sqrt function.
    """

    @staticmethod
    def forward(ctx: Any, input: Any, eps: float) -> torch.Tensor:  # type: ignore
        sqrt_input = torch.sqrt(input)
        ctx.save_for_backward(sqrt_input)
        ctx.eps = eps
        return sqrt_input

    @staticmethod
    def backward(ctx: Any, grad_output):  # type: ignore
        (input,) = ctx.saved_tensors
        return grad_output / (2 * (input + ctx.eps)), None


def sqrt_with_gradeps(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""
    Square-root of input with a "valid" gradient at 0.

    .. math::
        \frac{\partial f}{\partial x} = \frac{1}{2\sqrt{x}+\epsilon}

    Args:
        input: Tensor of arbitrary shape.
        eps: Value to add to the input when computing gradient (must be positive).

    Returns:
        A tensor whose value is the square-root of the input but whose associated
        autograd functions is :py:class:`SqrtEpsGrad`.
    """
    return SqrtEpsGrad.apply(input, eps)  # type: ignore
