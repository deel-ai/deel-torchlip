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

import torch
import torch.nn.functional as F


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


def group_sort(
    input: torch.Tensor, group_size: Optional[int] = None, dim: int = 1
) -> torch.Tensor:
    r"""
    Applies GroupSort activation on the given tensor.

    See Also:
        :py:func:`group_sort_2`
        :py:func:`full_sort`
    """

    if group_size is None or group_size > input.shape[dim]:
        group_size = input.shape[dim]

    if input.shape[dim] % group_size != 0:
        raise ValueError("The input size must be a multiple of the group size.")

    new_shape = (
        input.shape[:dim]
        + (input.shape[dim] // group_size, group_size)
        + input.shape[dim + 1 :]
    )
    if group_size == 2:
        resh_input = input.view(new_shape)
        a, b = (
            torch.min(resh_input, dim + 1, keepdim=True)[0],
            torch.max(resh_input, dim + 1, keepdim=True)[0],
        )
        return torch.cat([a, b], dim=dim + 1).view(input.shape)
    fv = input.reshape(new_shape)

    return torch.sort(fv, dim=dim + 1)[0].reshape(input.shape)


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


def kr_loss(input: torch.Tensor, target: torch.Tensor, multi_gpu=False) -> torch.Tensor:
    r"""
    Loss to estimate the Wasserstein-1 distance using Kantorovich-Rubinstein duality,
    as per

    .. math::
        \mathcal{W}(\mu, \nu) = \sup\limits_{f\in{}Lip_1(\Omega)}
            \underset{\mathbf{x}\sim{}\mu}{\mathbb{E}}[f(\mathbf{x})]
            - \underset{\mathbf{x}\sim{}\nu}{\mathbb{E}}[f(\mathbf{x})]

    where :math:`\mu` and :math:`\nu` are the distributions corresponding to the
    two possible labels as specific by their sign.

    `target` accepts label values in (0, 1), (-1, 1), or pre-processed with the
    `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

    Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
    pre-process the labels `target` with the
    `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        multi_gpu (bool): set to True when running on multi-GPU/TPU

    Returns:
        The Wasserstein-1 loss between ``input`` and ``target``.
    """
    if multi_gpu:
        return kr_loss_multi_gpu(input, target)
    else:
        return kr_loss_standard(input, target)


def kr_loss_standard(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""
    Loss to estimate the Wasserstein-1 distance using Kantorovich-Rubinstein duality,
    as per

    .. math::
        \mathcal{W}(\mu, \nu) = \sup\limits_{f\in{}Lip_1(\Omega)}
            \underset{\mathbf{x}\sim{}\mu}{\mathbb{E}}[f(\mathbf{x})]
            - \underset{\mathbf{x}\sim{}\nu}{\mathbb{E}}[f(\mathbf{x})]

    where :math:`\mu` and :math:`\nu` are the distributions corresponding to the
    two possible labels as specific by their sign.

    `target` accepts label values in (0, 1), (-1, 1)

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.

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
) -> torch.Tensor:
    """
    Loss to estimate the negative wasserstein-1 distance using Kantorovich-Rubinstein
    duality.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        multi_gpu (bool): set to True when running on multi-GPU/TPU

    Returns:
        The negative Wasserstein-1 loss between ``input`` and ``target``.

    See Also:
        :py:func:`kr_loss`
    """
    return -kr_loss(input, target, multi_gpu=multi_gpu)


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
) -> torch.Tensor:
    """
    Loss to estimate the wasserstein-1 distance with a hinge regularization using
    Kantorovich-Rubinstein duality.

    Args:
        input: Tensor of arbitrary shape.
        target: Tensor of the same shape as input.
        alpha: Regularization factor ([0,1]) between the hinge and the KR loss.
        min_margin: Minimal margin for the hinge loss.
        multi_gpu (bool): set to True when running on multi-GPU/TPU

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


def hinge_multiclass_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    min_margin: float = 1,
) -> torch.Tensor:
    """
    Loss to estimate the Hinge loss in a multiclass setup. It compute the
    elementwise hinge term. Note that this formulation differs from the
    one commonly found in tensorflow/pytorch (with maximise the difference
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
        alpha: Regularization factor ([0,1]) between the hinge and the KR loss.
        min_margin: Minimal margin for the hinge loss.
        multi_gpu (bool): set to True when running on multi-GPU/TPU

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


class SymmetricPad(torch.nn.Module):
    """
    Pads a 2D tensor symmetrically.

    Args:
        pad (tuple): A tuple (pad_left, pad_right, pad_top, pad_bottom) specifying
                 the number of pixels to pad on each side. (or single int if
                 common padding).

        onedim: False for conv2d, True for conv1d.

    """

    def __init__(self, pad, onedim=False):
        super().__init__()
        self.onedim = onedim
        num_dim = 2 if onedim else 4
        if isinstance(pad, int):
            self.pad = (pad,) * num_dim
        else:
            self.pad = torch.nn.modules.utils._reverse_repeat_tuple(pad, 2)
        assert len(self.pad) == num_dim, f"Pad must be a tuple of {num_dim} integers"

    def forward(self, x):

        # Horizontal padding
        left = x[:, ..., : self.pad[0]].flip(dims=[-1])
        right = x[:, ..., -self.pad[1] :].flip(dims=[-1])
        x = torch.cat([left, x, right], dim=-1)
        if self.onedim:
            return x
        # Vertical padding
        top = x[:, :, : self.pad[2], :].flip(dims=[-2])
        bottom = x[:, :, -self.pad[3] :, :].flip(dims=[-2])
        x = torch.cat([top, x, bottom], dim=-2)

        return x
