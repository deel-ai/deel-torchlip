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
from typing import Optional
from typing import Tuple
from typing import Union

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
        \text{LPReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \min(\max(a, -k), k) * x, & \text{ otherwise }
        \end{cases}

    """
    return F.prelu(input, torch.clamp(weight, -k_coef_lip, +k_coef_lip))


# Losses
def apply_reduction(val: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "auto":
        reduction = "mean"
    red = getattr(torch, reduction, None)
    if red is None:
        return val
    return red(val)


def kr_loss(
    input: torch.Tensor, target: torch.Tensor, multi_gpu=False, true_values=None
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
        true_values: depreciated (target>0 is used)

    Returns:
        The Wasserstein-1 loss between ``input`` and ``target``.
    """
    if multi_gpu:
        return kr_loss_multi_gpu(input, target)
    else:
        return kr_loss_standard(input, target)


def kr_loss_standard(
    input: torch.Tensor, target: torch.Tensor, true_values=None
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
        true_values: depreciated (target>0 is used)

    Returns:
        The Wasserstein-1 loss between ``input`` and ``target``.
    """

    target = target.view(input.shape)
    pos_target = (target > 0).to(input.dtype)
    mean_pos = torch.mean(pos_target, dim=0)
    # pos factor = batch_size/number of positive samples
    pos_factor = torch.nan_to_num(1.0 / mean_pos)
    # neg factor = batch_size/number of negative samples
    neg_factor = -torch.nan_to_num(1.0 / (1.0 - mean_pos))

    weighted_input = torch.where(target > 0, pos_factor, neg_factor) * input
    # Since element-wise KR terms are averaged by loss reduction later on, it is needed
    # to multiply by batch_size here.
    # In binary case (`y_true` of shape (batch_size, 1)), `tf.reduce_mean(axis=-1)`
    # behaves like `tf.squeeze()` to return element-wise loss of shape (batch_size, ).
    return torch.mean(weighted_input, dim=-1)


def kr_loss_multi_gpu(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Returns the element-wise KR loss when computing with a multi-GPU/TPU strategy.

    `target` and `input` can be either of shape (batch_size, 1) or
    (batch_size, # classes).

    When using this loss function, the labels `target` must be pre-processed with the
    `process_labels_for_multi_gpu()` function.

    Args:
        input: Tensor of arbitrary shape.
        target: pre-processed Tensor of the same shape as input.

    Returns:
        The Wasserstein-1 loss between ``input`` and ``target``.
    """
    target = target.view(input.shape).to(input.dtype)
    # Since the information of batch size was included in `target` by
    # `process_labels_for_multi_gpu()`, there is no need here to multiply by batch size.
    # In binary case (`target` of shape (batch_size, 1)), `torch.mean(dim=-1)`
    # behaves like `torch.squeeze()` to return element-wise loss of shape (batch_size,)
    return torch.mean(input * target, dim=-1)


def neg_kr_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    multi_gpu=False,
    true_values=None,
) -> torch.Tensor:
    """
    Loss to estimate the negative wasserstein-1 distance using Kantorovich-Rubinstein
    duality.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        true_values: depreciated (target>0 is used)

    Returns:
        The negative Wasserstein-1 loss between ``input`` and ``target``.

    See Also:
        :py:func:`kr_loss`
    """
    return -kr_loss(input, target, multi_gpu=multi_gpu)


# def neg_kr_loss_multi_gpu(
#     input: torch.Tensor,
#     target: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Loss to estimate the negative wasserstein-1 distance using Kantorovich-Rubinstein
#     duality.

#     `target` and `input` can be either of shape (batch_size, 1) or
#     (batch_size, # classes).

#     When using this loss function, the labels `target` must be pre-processed with the
#     `process_labels_for_multi_gpu()` function.

#     Args:
#         input: Tensor of arbitrary shape.
#         target: pre-processed Tensor of the same shape as input.

#     Returns:
#         The negative Wasserstein-1 loss between ``input`` and ``target``.

#     See Also:
#         :py:func:`kr_loss`
#     """
#     return -kr_loss_multi_gpu(input, target)


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
    sign_target = torch.where(target > 0, 1.0, -1.0).to(input.dtype)
    hinge = F.relu(min_margin / 2.0 - sign_target * input)
    return torch.mean(hinge, dim=-1)


def hkr_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    min_margin: float = 1.0,
    multi_gpu=False,
    true_values=None,
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
    kr_loss_fct = kr_loss_multi_gpu if multi_gpu else kr_loss
    assert alpha <= 1.0
    if alpha == 1.0:  # alpha for  hinge only
        return hinge_margin_loss(input, target, min_margin)
    if alpha == 0:
        return -kr_loss_fct(input, target)
    # true value: positive value should be the first to be coherent with the
    # hinge loss (positive y_pred)
    return alpha * hinge_margin_loss(input, target, min_margin) - (
        1 - alpha
    ) * kr_loss_fct(input, target)


# def hkr_loss_multi_gpu(
#     input: torch.Tensor,
#     target: torch.Tensor,
#     alpha: float,
#     min_margin: float = 1.0,
# ) -> torch.Tensor:
#     """
#     Loss to estimate the wasserstein-1 distance with a hinge regularization using
#     Kantorovich-Rubinstein duality.

#     Args:
#         input: Tensor of arbitrary shape.
#         target: Tensor of the same shape as input.
#         alpha: Regularization factor between the hinge and the KR loss.
#         min_margin: Minimal margin for the hinge loss.
#         true_values: tuple containing the two label for each predicted class.

#     Returns:
#         The regularized Wasserstein-1 loss.

#     See Also:
#         :py:func:`hinge_margin_loss`
#         :py:func:`kr_loss`
#     """
#     assert alpha <= 1.0
#     if alpha == 1.0:  # alpha for  hinge only
#         return hinge_margin_loss(input, target, min_margin)
#     if alpha == 0:
#         return -kr_loss_multi_gpu(input, target)
#     # true value: positive value should be the first to be coherent with the
#     # hinge loss (positive y_pred)
#     return alpha * hinge_margin_loss(input, target, min_margin) - (
#         1 - alpha
#     ) * kr_loss_multi_gpu(input, target)


# def kr_multiclass_loss(
#     input: torch.Tensor,
#     target: torch.Tensor,
# ) -> torch.Tensor:
#     r"""
#     Loss to estimate average of W1 distance using Kantorovich-Rubinstein
#     duality over outputs. In this multiclass setup thr KR term is computed
#     for each class and then averaged.

#     Args:
#         input: Tensor of arbitrary shape.
#         target: Tensor of the same shape as input.
#                 target has to be one hot encoded (labels being 1s and 0s ).

#     Returns:
#         The Wasserstein multiclass loss between ``input`` and ``target``.
#     """
#     return kr_loss(input, target)
#     # true_target = torch.where(target > 0, 1.0, 0.0).to(input.dtype)
#     # esp_true_true = torch.sum(input * true_target, 0) / torch.sum(true_target, 0)
#     # esp_false_true = torch.sum(input * (1 - true_target), 0) / torch.sum(
#     #     (1 - true_target), 0
#     # )

#     # return torch.mean(esp_true_true - esp_false_true)


def hinge_multiclass_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    min_margin: float = 1,
) -> torch.Tensor:
    """
    Loss to estimate the Hinge loss in a multiclass setup. It compute the
    elementwise hinge term. Note that this formulation differs from the
    one commonly found in tensorflow/pytorch (with marximise the difference
    between the two largest logits). This formulation is consistent with the
    binary classification loss used in a multiclass fashion.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input containing
            one hot encoding target labels (0 and +1).
        min_margin: The minimal margin to enforce.
    Note:
        target should be one hot encoded. labels in (1,0)

    Returns:
        The hinge margin multiclass loss.
    """
    sign_target = torch.where(target > 0, 1.0, -1.0).to(input.dtype)
    hinge = F.relu(min_margin / 2.0 - sign_target * input)
    # reweight positive elements
    factor = target.shape[-1] - 1.0
    hinge = torch.where(target > 0, hinge * factor, hinge)
    return torch.mean(hinge, dim=-1)


def hkr_multiclass_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.0,
    min_margin: float = 1.0,
    multi_gpu=False,
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

    assert alpha <= 1.0
    kr_loss_fct = kr_loss_multi_gpu if multi_gpu else kr_loss
    if alpha == 1.0:  # alpha  hinge only
        return hinge_multiclass_loss(input, target, min_margin)
    elif alpha == 0.0:  # alpha = 0 => KR only
        return -kr_loss_fct(input, target)
    else:
        return alpha * hinge_multiclass_loss(input, target, min_margin) - (
            1 - alpha
        ) * kr_loss_fct(input, target)


# def hkr_multiclass_loss_multi_gpu(
#     input: torch.Tensor,
#     target: torch.Tensor,
#     alpha: float = 0.0,
#     min_margin: float = 1.0,
# ) -> torch.Tensor:
#     """
#     Loss to estimate the wasserstein-1 distance with a hinge regularization using
#     Kantorovich-Rubinstein duality.

#     Args:
#         input: Tensor of arbitrary shape.
#         target: Tensor of the same shape as input.
#         alpha: Regularization factor between the hinge and the KR loss.
#         min_margin: Minimal margin for the hinge loss.
#         true_values: tuple containing the two label for each predicted class.

#     Returns:
#         The regularized Wasserstein-1 loss.

#     See Also:
#         :py:func:`hinge_margin_loss`
#         :py:func:`kr_loss`
#     """

#     assert alpha <= 1.0
#     if alpha == 1.0:  # alpha  hinge only
#         return hinge_multiclass_loss(input, target, min_margin)
#     elif alpha == 0.0:  # alpha = 0 => KR only
#         return -kr_loss_multi_gpu(input, target)
#     else:
#         return alpha * hinge_multiclass_loss(input, target, min_margin) - (
#             1 - alpha
#         ) * kr_loss_multi_gpu(input, target)


def process_labels_for_multi_gpu(labels: torch.Tensor) -> torch.Tensor:
    """Process labels to be fed to any loss based on KR estimation with a multi-GPU/TPU
    strategy.

    When using a multi-GPU/TPU strategy, the flag `multi_gpu` in KR-based losses must be
    set to True and the labels have to be pre-processed with this function.

    For binary classification, the labels should be of shape [batch_size, 1].
    For multiclass problems, the labels must be one-hot encoded (1 or 0) with shape
    [batch_size, number of classes].

    Args:
        labels (torch.Tensor): tensor containing the labels

    Returns:
        torch.Tensor: labels processed for KR-based losses with multi-GPU/TPU strategy.
    """
    pos_labels = torch.where(labels > 0, 1.0, 0.0).to(labels.dtype)
    mean_pos = torch.mean(pos_labels, dim=0)
    # pos factor = batch_size/number of positive samples
    pos_factor = torch.nan_to_num(1.0 / mean_pos)
    # neg factor = batch_size/number of negative samples
    neg_factor = -torch.nan_to_num(1.0 / (1.0 - mean_pos))

    # Since element-wise KR terms are averaged by loss reduction later on, it is needed
    # to multiply by batch_size here.
    return torch.where(labels > 0, pos_factor, neg_factor)
