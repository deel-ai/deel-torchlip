# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Contains utility functions.
"""

from typing import Any

import numpy as np
import torch
import torch.autograd

from torch.nn import Sequential

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
    return SqrtEpsGrad.apply(input, eps)  # type: ignore


def evaluate_lip_const(model: Sequential, x, eps=1e-4, seed=None):
    """
    Evaluate the Lipschitz constant of a model, with the naive method.
    Please note that the estimation of the lipschitz constant is done locally around
    input sample. This may not correctly estimate the behaviour in the whole domain.

    Args:
        model: built torch model used to make predictions
        x: inputs used to compute the lipschitz constant
        eps: magnitude of noise to add to input in order to compute the constant
        seed: seed used when generating the noise ( can be set to None )

    Returns:
        the empirically evaluated lipschitz constant. The computation might also be
        inaccurate in high dimensional space.

    """
    y_pred = model(x.float())
    np.random.seed(seed)
    x_var = x + torch.from_numpy(
        np.random.uniform(low=eps * 0.25, high=eps, size=x.shape)
    )
    y_pred_var = model(x_var.float())
    dx = x - x_var
    dfx = y_pred - y_pred_var
    ndx = torch.sum(torch.square(dx), dim=list(range(1, len(x.shape))))
    ndfx = torch.sum(torch.square(dfx.data), dim=list(range(1, len(y_pred.shape))))
    lip_cst = torch.sqrt(torch.max(ndfx / ndx))
    print("lip cst: %.3f" % lip_cst)
    # try:
    #     layer = model[0]
    #     u, s, v = torch.svd(layer.weight_orig)
    #     actual_orig_sn = s.mean()
    #     orig_sn = (layer.weight_u @ layer.weight_orig @ layer.weight_v).item()
    #     print("Original spectral norm:              ", actual_orig_sn)
    #     print("Approximate original spectral norm:  ", orig_sn)

    #     # updated weights singular values
    #     u, new_sn, v = torch.svd(layer.weight.data, compute_uv=False)
    #     print(new_sn)
    #     updated_sn = new_sn.mean()
    #     print("updated spectral norm:", updated_sn)
    # except ValueError:
    #     pass
    return lip_cst


def compute_lconv_ip_coef(kernel_size, input_shape=None, stride=1):
    # According to the file lipschitz_CNN.pdf
    stride = np.prod(stride)
    k1 = kernel_size[0]
    k1_div2 = (k1 - 1) / 2
    k2 = kernel_size[1]
    k2_div2 = (k2 - 1) / 2
    h = input_shape[1]
    w = input_shape[2]
    if stride == 1:
        coefLip = np.sqrt(
            (w * h)
            / ((k1 * h - k1_div2 * (k1_div2 + 1)) * (k2 * w - k2_div2 * (k2_div2 + 1)))
        )
    else:
        sn1 = stride[0]
        sn2 = stride[1]
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
