# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn.functional as F

# Up / Down sampling


def invertible_downsample(
    input: torch.Tensor, kernel_size: Union[int, Tuple[int, ...]]
) -> torch.Tensor:
    """
    Downsamples the input in an invertible way.

    The number of elements in the output tensor is the same as the
    number of elements in the input tensor.

    Args:
        input: A tensor of shape :math:`(N, C, W)`, :math:`(N, C, W, H)` or
            :math:`(N, C, D, W, H)` to downsample.
        kernel_size: The downsample scale. If a single-value is passed, the
            same value will be used alongside all dimensions, otherwise the
            length of ``kernel_size`` must match the number of dimensions
            of the input (1, 2 or 3).

    Raises:
        ValueError: If there is a mismatch between ``kernel_size`` and the input
            shape.

    Examples:

        >>> x = torch.rand(16, 16, 32, 32)
        >>> x.shape
        (16, 16, 32, 32)
        >>> y = invertible_downsample(x, (2, 4))
        >>> y.shape
        (16, 128, 16, 8)

    See Also:
        :py:func:`invertible_upsample`
    """

    # number of dimensions
    shape = input.shape
    ndims = len(shape) - 2
    ks: Tuple[int, ...]
    if isinstance(kernel_size, int):
        ks = (kernel_size,) * ndims
    else:
        ks = tuple(kernel_size)

    if len(ks) != ndims:
        raise ValueError(
            f"Expected {len(ks) + 2}-dimensional input for kernel size {ks}, but "
            f"got {ndims + 2}-dimensional input of size {shape} instead"
        )

    for i in range(ndims):
        input = input.reshape(
            input.shape[: 2 + 2 * i]
            + (ks[i], shape[i + 2] // ks[i])
            + input.shape[2 + 2 * i + 1 :]
        )

    # order of the permutation
    perm = (
        [0, 1]
        + [2 + 2 * i for i in range(ndims)]
        + [2 + 2 * i + 1 for i in range(ndims)]
    )
    input = input.permute(*perm)

    return input.reshape(input.shape[:1] + (-1,) + input.shape[-ndims:])


def invertible_upsample(
    input: torch.Tensor, kernel_size: Union[int, Tuple[int, ...]]
) -> torch.Tensor:
    r"""
    Upsamples the input in an invertible way. The number of elements in the
    output tensor is the same as the number of elements in the input tensor.

    The number of input channels must be a multiple of the product of the
    kernel sizes, i.e.

    .. math::
        C \equiv 0 \mod (k_1 * \ldots{} * k_l)

    where :math:`C` is the number of inputs channels and :math:`k_i` the kernel
    size for dimension :math:`i` and :math:`l` the number of dimensions.

    Args:
        input: A tensor of shape :math:`(N, C, W)`, :math:`(N, C, W, H)` or
            :math:`(N, C, D, W, H)` to upsample.
        kernel_size: The upsample scale. If a single-value is passed, the
            same value will be used alongside all dimensions, otherwise the
            length of ``kernel_size`` must match the number of dimensions
            of the input (1, 2 or 3).

    Raises:
        ValueError: If there is a mismatch between ``kernel_size`` and the input
            shape.

    Examples:

        >>> x = torch.rand(16, 128, 16, 8)
        >>> x.shape
        (16, 128, 16, 8)
        >>> y = invertible_upsample(x, (2, 4))
        >>> y.shape
        (16, 16, 32, 32)

    See Also:
        :py:func:`invertible_downsample`
    """

    # number of dimensions
    ndims = len(input.shape) - 2
    ks: Tuple[int, ...]
    if isinstance(kernel_size, int):
        ks = (kernel_size,) * ndims
    else:
        ks = tuple(kernel_size)

    if len(ks) != ndims:
        raise ValueError(
            f"Expected {len(ks) + 2}-dimensional input for kernel size {ks}, but "
            f"got {ndims + 2}-dimensional input of size {input.shape} instead"
        )

    ncout = input.shape[1] // np.prod(ks)
    input = input.reshape((-1, ncout) + ks + input.shape[-ndims:])

    # order of the permutation
    perm = [0, 1]
    for i in range(ndims):
        perm += [2 + i, 2 + i + ndims]
    input = input.permute(*perm)

    for i in range(ndims):
        input = input.reshape(input.shape[: 2 + i] + (-1,) + input.shape[2 + i + 2 :])

    return input


# Activations


def max_min(input: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    r"""
    Applies max-min activation on the given tensor.

    If ``input`` is a tensor of shape :math:`(N, C)` and ``dim`` is
    ``None``, the output can be described as:

    .. math::
        \text{out}(N_i, C_{2j}) = \max(\text{input}(N_i, C_j), 0)\\
        \text{out}(N_i, C_{2j + 1}) = \max(-\text{input}(N_i, C_j), 0)

    where :math:`N` is the batch size and :math:`C` is the size of the
    tensor.

    Args:
        input:  A tensor of arbitrary shape.
        dim: The dimension to apply max-min. If None, will apply to the
            0th dimension if the shape of input is :math:`(C)` or to the
            first if its :math:`(N, C, *)`.

    Returns:
        A tensor of shape :math:`(2C)` or :math:`(N, 2C, *)` depending
        on the shape of the input.

    Note:
        M. Blot, M. Cord, et N. Thome, « Max-min convolutional neural networks
        for image classification », in 2016 IEEE International Conference on Image
        Processing (ICIP), Phoenix, AZ, USA, 2016, p. 3678‑3682.
    """
    if dim is None:
        if len(input.shape) == 1:
            dim = 0
        else:
            dim = 1
    return torch.cat((F.relu(input), F.relu(-input)), dim=dim)


def group_sort(input: torch.Tensor, group_size: Optional[int] = None) -> torch.Tensor:
    r"""
    Applies GroupSort activation on the given tensor.

    See Also:
        :py:func:`group_sort_2`
        :py:func:`full_sort`
    """
    if group_size is None or group_size > input.shape[1]:
        group_size = input.shape[1]

    if input.shape[1] % group_size != 0:
        raise ValueError("The input size must be a multiple of the group size.")

    fv = input.reshape([-1, group_size])
    if group_size == 2:
        sfv = torch.chunk(fv, 2, 1)
        b = sfv[0]
        c = sfv[1]
        newv = torch.cat((torch.min(b, c), torch.max(b, c)), dim=1)
        newv = newv.reshape(input.shape)
        return newv

    return torch.sort(fv)[0].reshape(input.shape)


def group_sort_2(input: torch.Tensor) -> torch.Tensor:
    r"""
    Applies GroupSort-2 activation on the given tensor. This function is equivalent
    to ``group_sort(input, 2)``.

    See Also:
        :py:func:`group_sort`
    """
    return group_sort(input, 2)


def full_sort(input: torch.Tensor) -> torch.Tensor:
    r"""
    Applies FullSort activation on the given tensor. This function is equivalent
    to ``group_sort(input, None)``.

    See Also:
        :py:func:`group_sort`
    """
    return group_sort(input, None)


def lipschitz_prelu(
    input: torch.Tensor, weight: torch.Tensor, k_coef_lip: float = 1.0
) -> torch.Tensor:
    r"""
    Applies k-Lipschitz version of PReLU by clamping the weights

    .. math::
        \text{LipschitzPReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \min(\max(a, -k), k) * x, & \text{ otherwise }
        \end{cases}

    """
    return F.prelu(input, torch.clamp(weight, -k_coef_lip, +k_coef_lip))


# Losses


def kr_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    true_values: Tuple[int, int] = (0, 1),
) -> torch.Tensor:
    r"""
    Loss to estimate the Wasserstein-1 distance using Kantorovich-Rubinstein duality,
    as per

    .. math::
        \mathcal{W}(\mu, \nu) = \sup\limits_{f\in{}Lip_1(\Omega)}
            \underset{\mathbf{x}\sim{}\mu}{\mathbb{E}}[f(\mathbf{x})]
            - \underset{\mathbf{x}\sim{}\nu}{\mathbb{E}}[f(\mathbf{x})]

    where :math:`\mu` and :math:`\nu` are the distributions corresponding to the
    two possible labels as specific by ``true_values``.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        true_values: Tuple containing the two label for the predicted class.

    Returns:
        The Wasserstein-1 loss between ``input`` and ``target``.
    """

    v0, v1 = true_values
    target = target.view(input.shape)
    return torch.mean(input[target == v0]) - torch.mean(input[target == v1])


def neg_kr_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    true_values: Tuple[int, int] = (0, 1),
) -> torch.Tensor:
    """
    Loss to estimate the negative wasserstein-1 distance using Kantorovich-Rubinstein
    duality.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        true_values: Tuple containing the two label for the predicted classes.

    Returns:
        The negative Wasserstein-1 loss between ``input`` and ``target``.

    See Also:
        :py:func:`kr_loss`
    """
    return -kr_loss(input, target, true_values)


def hinge_margin_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    min_margin: float = 1,
) -> torch.Tensor:
    r"""
    Compute the hinge margin loss as per

    .. math::
        \underset{\mathbf{x}}{\mathbb{E}}
        [\max(0, 1 - \mathbf{y} f(\mathbf{x}))]

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input containing
            target labels (-1 and +1).
        min_margin: The minimal margin to enforce.

    Returns:
        The hinge margin loss.
    """
    target = target.view(input.shape)
    return torch.mean(
        torch.max(
            torch.zeros_like(input),
            min_margin - torch.sign(target) * input,
        )
    )


def hkr_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    min_margin: float = 1.0,
    true_values: Tuple[int, int] = (-1, 1),
) -> torch.Tensor:
    """
    Loss to estimate the wasserstein-1 distance with a hinge regularization using
    Kantorovich-Rubinstein duality.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        alpha: Regularization factor between the hinge and the KR loss.
        min_margin: Minimal margin for the hinge loss.
        true_values: tuple containing the two label for each predicted class.

    Returns:
        The regularized Wasserstein-1 loss.

    See Also:
        :py:func:`hinge_margin_loss`
        :py:func:`kr_loss`
    """
    if alpha == np.inf:  # alpha negative hinge only
        return hinge_margin_loss(input, target, min_margin)

    # true value: positive value should be the first to be coherent with the
    # hinge loss (positive y_pred)
    return alpha * hinge_margin_loss(input, target, min_margin) - kr_loss(
        input, target, (true_values[1], true_values[0])
    )
