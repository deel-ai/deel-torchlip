# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Contains utility functions.
"""

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.autograd

DEFAULT_NITER_BJORCK = 15
DEFAULT_NITER_SPECTRAL = 3
DEFAULT_NITER_SPECTRAL_INIT = 10


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
    """
    Square-root of input with a valid gradient at 0.

    Args:
        input: Tensor of arbitrary shape.
        eps: Value to add to the input when computing gradient (must be positive).

    Returns:
        A tensor whose value is the square-root of the input but whose associated
        autograd functions is `SqrtEpsGrad`.
    """
    return SqrtEpsGrad.apply(input, eps)  # type: ignore


def evaluate_lip_const(
    model: torch.nn.Module,
    x: torch.Tensor,
    eps: float = 1e-4,
    seed: Optional[int] = None,
) -> float:
    """
    Evaluate the Lipschitz constant of a model, with the naive method.
    Please note that the estimation of the lipschitz constant is done locally around
    input sample. This may not correctly estimate the behaviour in the whole domain.

    Args:
        model: Torch model used to make predictions.
        x: Inputs used to compute the lipschitz constant.
        eps: Magnitude of noise to add to input in order to compute the constant.
        seed: Seed used when generating the noise. If None, a random seed will be
            used.

    Returns:
        The empirically evaluated lipschitz constant. The computation might also be
        inaccurate in high dimensional space.

    """
    y_pred = model(x.float())

    with torch.random.fork_rng():
        if seed is None:
            torch.random.seed()
        else:
            torch.random.manual_seed(seed)
        x_var = x + torch.distributions.Uniform(low=eps * 0.25, high=eps).sample(
            x.shape
        )

    y_pred_var = model(x_var.float())

    dx = x - x_var
    dfx = y_pred - y_pred_var
    ndx = torch.sum(torch.square(dx), dim=tuple(range(1, len(x.shape))))
    ndfx = torch.sum(torch.square(dfx.data), dim=tuple(range(1, len(y_pred.shape))))
    lip_cst = torch.sqrt(torch.max(ndfx / ndx))

    return lip_cst.item()


def compute_lconv_ip_coef(
    kernel_size: Tuple[int, int],
    input_shape: Tuple[int, ...],
    strides: Tuple[int, int] = (1, 1),
) -> float:
    # According to the file lipschitz_CNN.pdf
    stride = np.prod(strides)
    k1, k2 = kernel_size
    h, w = input_shape[-2:]

    k1_div2 = (k1 - 1) / 2
    k2_div2 = (k2 - 1) / 2

    if stride == 1:
        coefLip = np.sqrt(
            (w * h)
            / ((k1 * h - k1_div2 * (k1_div2 + 1)) * (k2 * w - k2_div2 * (k2_div2 + 1)))
        )
    else:
        sn1 = strides[0]
        sn2 = strides[1]
        ho = np.floor(h / sn1)
        wo = np.floor(w / sn2)
        alphabar1 = np.floor(k1_div2 / sn1)
        alphabar2 = np.floor(k2_div2 / sn2)
        betabar1 = k1_div2 - alphabar1 * sn1
        betabar2 = k2_div2 - alphabar2 * sn2
        zl1 = (alphabar1 * sn1 + 2 * betabar1) * (alphabar1 + 1) / 2
        zl2 = (alphabar2 * sn2 + 2 * betabar2) * (alphabar2 + 1) / 2
        gamma1 = h - 1 - sn1 * np.ceil((h - 1 - k1_div2) / sn1)
        gamma2 = w - 1 - sn2 * np.ceil((w - 1 - k2_div2) / sn2)
        alphah1 = np.floor(gamma1 / sn1)
        alphaw2 = np.floor(gamma2 / sn2)
        zr1 = (alphah1 + 1) * (k1_div2 - gamma1 + sn1 * alphah1 / 2.0)
        zr2 = (alphaw2 + 1) * (k2_div2 - gamma2 + sn2 * alphaw2 / 2.0)
        coefLip = np.sqrt((h * w) / ((k1 * ho - zl1 - zr1) * (k2 * wo - zl2 - zr2)))

    return coefLip
