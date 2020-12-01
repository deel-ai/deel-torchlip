# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Contains utility functions.
"""

import numpy as np
import torch
from torch.nn import Sequential

DEFAULT_NITER_BJORCK = 15
DEFAULT_NITER_SPECTRAL = 3
DEFAULT_NITER_SPECTRAL_INIT = 10

CUSTOM_OBJECTS = dict()


def _deel_export(f):
    """
    Annotation, allows to automatically add deel custom objects to the
    deel.lip.utils.CUSTOM_OBJECTS variable, which is useful when working with custom
    layers.
    """
    global CUSTOM_OBJECTS
    CUSTOM_OBJECTS[f.__name__] = f
    return f


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
    # x = np.repeat(x, 100, 0)
    # y_pred = np.repeat(y_pred, 100, 0)
    np.random.seed(seed)
    x_var = x + torch.from_numpy(
        np.random.uniform(low=eps * 0.25, high=eps, size=x.shape)
    )
    # x_var = x + FloatTensor(shape=x.shape)).uniform_(eps * 0.25, eps, seed=seed)
    y_pred_var = model(x_var.float())
    dx = x - x_var
    dfx = y_pred - y_pred_var
    ndx = torch.sum(torch.square(dx), axis=list(range(1, len(x.shape))))
    ndfx = torch.sum(torch.square(dfx.data), axis=list(range(1, len(y_pred.shape))))
    lip_cst = torch.sqrt(torch.max(ndfx / ndx))
    print("lip cst: %.3f" % lip_cst)
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


def bjorck_normalization(w, niter=DEFAULT_NITER_BJORCK):
    """
    apply Bjorck normalization on w.

    Args:
        w: weight to normalize, in order to work properly, we must have
            max_eigenval(w) ~= 1
        niter: number of iterations

    Returns:
        the orthonormal weights

    """
    shape = w.shape
    height = w.size(0)
    w_mat = w.reshape(height, -1)
    for i in range(niter):
        w_mat = 1.5 * w_mat - 0.5 * torch.mm(w_mat, torch.mm(w_mat.t(), w_mat))
    w = w_mat.reshape(shape)
    return w
