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
        ctx.save_for_backward(input)
        ctx.eps = eps
        return torch.sqrt(input)

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
