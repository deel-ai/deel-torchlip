# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains computation function, for BjorckNormalizer and spectral
normalization. This is done for internal use only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_NITER_BJORCK = 15
DEFAULT_NITER_SPECTRAL = 3
DEFAULT_NITER_SPECTRAL_INIT = 10
DEFAULT_BETA = 0.5


def bjorck_normalization(w: torch.Tensor, niter: int = DEFAULT_NITER_BJORCK):
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
        # W = tf.Print(W,[tf.shape(W)])
        w = (1.0 + DEFAULT_BETA) * w_mat - DEFAULT_BETA * torch.mm(w_mat, torch.mm(w_mat.t(), w_mat))
    w = w_mat.reshape(shape)
    return w


def _power_iteration(
    w: torch.Tensor, u: torch.Tensor, niter: int = DEFAULT_NITER_SPECTRAL
):
    """
    Internal function that performs the power iteration algorithm.

    Args:
        w: weights matrix that we want to find eigen vector
        u: initialization of the eigen vector
        niter: number of iteration, must be greater than 0

    Returns:
         u and v corresponding to the maximum eigenvalue

    """
    _u = u
    for i in range(niter):
        _v = F.normalize(torch.mm(_u, w.t()), p=2, dim=1)
        _u = F.normalize(torch.mm(_v, w), p=2, dim=1)
    return _u, _v


def spectral_normalization(
    kernel: torch.Tensor, u: torch.Tensor = None, niter: int = DEFAULT_NITER_SPECTRAL,
):
    """
    Normalize the kernel to have it's max eigenvalue == 1.

    Args:
        kernel: the kernel to normalize
        u: initialization for the max eigen vector
        niter: number of iteration

    Returns:
        the normalized kernel w_bar, it's shape, the maximum eigen vector, and the
        maximum eigen value

    """

    # Flatten the Tensor
    W_flat = kernel.flatten(start_dim=1)

    if u is None:
        niter *= 2  # if u was not double number of iterations for the first run
        # TODO : check K.random_normal => init.kaiming_normal
        u = torch.ones(tuple([1, W_flat.shape[-1]]), device=kernel.device)
        torch.nn.init.kaiming_normal_(u)

    # do power iteration
    _u, _v = _power_iteration(W_flat, u, niter)

    # Calculate Sigma
    sigma = _v.mm(W_flat).mm(_u.t())

    # Normalize W_bar
    W_bar = kernel.div(sigma)

    return W_bar, u, sigma


def qr_normalization(
    kernel: torch.Tensor, u: torch.Tensor = None, niter: int = DEFAULT_NITER_SPECTRAL,
):
    """ Make the kernel tensor 1-Lipschitz Using QR
    Currently not used and to be checked 
    """
    # TODO: check and fix the code. An ugly approximation is done using padding to have "square" tensors
    # TODO: this approximation has to be checked and fixed if possible

    # Pad weight tensor to be square, cubic, etc..
    max_dim = max(kernel.shape[:-1])
    pad_size = []
    for dim in kernel.shape:
        pad_size = [0, max_dim - dim] + pad_size

    # Pad with 0.0 to have a "square" tensor for qr
    pad = nn.ConstantPad2d(pad_size, 0.0)
    padded_weight = pad(module.weight)

    # decompose weight into Q and R
    q, r = padded_weight.qr(some=False)

    # extract sub tensor from Q
    indices = [slice(0, dim) for dim in kernel.shape]

    return q[indices].to(device=kernel.device)


def make_weight_1lipschitz(module):
    if not hasattr(module, "lipschitz_u"):
        module.register_buffer("lipschitz_u", None)
    # print("weigth before", module.weight.shape)
    # print("CUDA mem befor", torch.cuda.memory_allocated())
    W_bar, module.lipschitz_u, sigma = spectral_normalization(
        module, module.lipschitz_u
    )
    module.weight.data = W_bar
    # module.weight = torch.nn.Parameter(W_bar)
    module.zero_grad()
    # # print("CUDA mem after", torch.cuda.memory_allocated())
    # # print("weigth after", module.weight.shape)
