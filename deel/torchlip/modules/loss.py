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
import warnings
from typing import Optional
import torch
from .. import functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch import Tensor


class KRLoss(torch.nn.Module):
    def __init__(self, multi_gpu=False, reduction: str = "mean", true_values=None):
        r"""
        Loss that estimates the Wasserstein-1 distance using the Kantorovich-Rubinstein
        duality.
        The Kantorovich-Rubinstein duality is formulated as following:

        $$
        W_1(\mu, \nu) =
        \sup_{f \in Lip_1(\Omega)} \underset{\textbf{x} \sim \mu}{\mathbb{E}}
        \left[f(\textbf{x} )\right] -
        \underset{\textbf{x}  \sim \nu}{\mathbb{E}} \left[f(\textbf{x} )\right]
        $$

        Where mu and nu stands for the two distributions, the distribution where the
        label is 1 and the rest.

        Note that `input` and `target` must be of rank 2: (batch_size, 1) or
        (batch_size, C) for multilabel classification (with C categories).
        `target` accepts label values in (0, 1), (-1, 1), or pre-processed with the
        `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `target` with the
        `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

        Args:
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: type of reduction applied to the output. possible values are
                'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
            true_values: depreciated.
        """
        super().__init__()
        self.reduction = reduction
        self.multi_gpu = multi_gpu
        assert (
            true_values is None
        ), "depreciated true_values should be None (use target>0)"

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_batch = F.kr_loss(input, target, multi_gpu=self.multi_gpu)
        return F.apply_reduction(loss_batch, self.reduction)


class NegKRLoss(torch.nn.Module):

    def __init__(self, multi_gpu=False, reduction: str = "mean", true_values=None):
        """
        Loss that estimates the negative of the Wasserstein-1 distance using
        the Kantorovich-Rubinstein duality. See `KRLoss` for more details.

        Args:
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: type of reduction applied to the output. possible values are
                'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
            true_values: depreciated.
        """
        super().__init__()
        self.reduction = reduction
        self.multi_gpu = multi_gpu
        assert (
            true_values is None
        ), "depreciated true_values should be None (use target>0)"

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_batch = F.neg_kr_loss(input, target, multi_gpu=self.multi_gpu)
        return F.apply_reduction(loss_batch, self.reduction)


class HingeMarginLoss(torch.nn.Module):
    def __init__(self, min_margin: float = 1.0, reduction: str = "mean"):
        """
        Hinge margin loss.

        Args:
            min_margin: The minimal margin to enforce.
            reduction: type of reduction applied to the output. possible values are
                'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
        """
        super().__init__()
        self.reduction = reduction
        self.min_margin = min_margin

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_batch = F.hinge_margin_loss(input, target, self.min_margin)
        return F.apply_reduction(loss_batch, self.reduction)


class HKRLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float,
        min_margin: float = 1.0,
        multi_gpu=False,
        reduction: str = "mean",
        true_values=None,
    ):
        """
        Loss that estimates the Wasserstein-1 distance using the Kantorovich-Rubinstein
        duality with a hinge regularization.

        [1] M. Serrurier, F. Mamalet, et al. «Achieving robustness in classification
        using optimal transport with hinge regularization», 2021.

        Note that `input` and `target` must be of rank 2: (batch_size, 1) or
        (batch_size, C) for multilabel classification (with C categories).
        `target` accepts label values in (0, 1), (-1, 1), or pre-processed with the
        `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `target` with the
        `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

        the regularization factor `alpha` is a value between 0 and 1. It controls the
        trade-off between the hinge and the KR loss. When `alpha` is 0, the loss is
        equivalent to the KR loss, and when `alpha` is 1, the loss is equivalent to the
        hinge loss.

        Args:
            alpha: Regularization factor ([0,1]) between the hinge and the KR loss.
            min_margin: Minimal margin for the hinge loss.
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: type of reduction applied to the output. possible values are
                'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
            true_values: depreciated.
        """
        super().__init__()
        self.reduction = reduction
        self.multi_gpu = multi_gpu
        if (alpha >= 0) and (alpha <= 1):
            self.alpha = alpha
        else:
            warnings.warn(
                f"Depreciated alpha should be between 0 and 1 replaced by \
                    {alpha/(alpha+1.0)}"
            )
            self.alpha = alpha / (alpha + 1.0)
        self.min_margin = min_margin
        assert (
            true_values is None
        ), "depreciated true_values should be None (use target>0)"

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_batch = F.hkr_loss(
            input, target, self.alpha, self.min_margin, multi_gpu=self.multi_gpu
        )
        return F.apply_reduction(loss_batch, self.reduction)


class KRMulticlassLoss(torch.nn.Module):
    def __init__(self, multi_gpu=False, reduction: str = "mean"):
        r"""
        Loss to estimate average of Wasserstein-1 distance using Kantorovich-Rubinstein
        duality over outputs. In this multiclass setup, the KR term is computed for each
        class and then averaged.

        Note that`target` should be one-hot encoded or pre-processed with the
        `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `target` with the
        `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

        Args:
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: type of reduction applied to the output. possible values are
                'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
        """
        super().__init__()
        self.reduction = reduction
        self.multi_gpu = multi_gpu

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_batch = F.kr_loss(input, target, multi_gpu=self.multi_gpu)
        return F.apply_reduction(loss_batch, self.reduction)


class HingeMulticlassLoss(torch.nn.Module):
    def __init__(self, min_margin: float = 1.0, reduction: str = "mean"):
        r"""
        Loss to estimate the Hinge loss in a multiclass setup. It computes the
        element-wise hinge term. Note that this formulation differs from the
        one commonly found in tensorflow/pytorch (with maximise the difference
        between the two largest logits). This formulation is consistent with the
        binary classification loss used in a multiclass fashion.

        Args:
            min_margin: The minimal margin to enforce.
            reduction: type of reduction applied to the output. possible values are
                'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
        """
        super().__init__()
        self.min_margin = min_margin
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_batch = F.hinge_multiclass_loss(input, target, self.min_margin)
        return F.apply_reduction(loss_batch, self.reduction)


class HKRMulticlassLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float,
        min_margin: float = 1.0,
        multi_gpu=False,
        reduction: str = "mean",
    ):
        """
        Loss that estimates the Wasserstein-1 distance using the Kantorovich-Rubinstein
        duality with a hinge regularization.

        [1] M. Serrurier, F. Mamalet, et al. «Achieving robustness in classification
        using optimal transport with hinge regularization», 2021.

        Note that`target` should be one-hot encoded or pre-processed with the
        `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `target` with the
        `deel.torchlip.functional.process_labels_for_multi_gpu()` function.

        the regularization factor `alpha` is a value between 0 and 1. It controls the
        trade-off between the hinge and the KR loss. When `alpha` is 0, the loss is
        equivalent to the KR loss, and when `alpha` is 1, the loss is equivalent to the
        hinge loss.

        Args:
            alpha: Regularization factor ([0,1]) between the hinge and the KR loss.
            min_margin: Minimal margin for the hinge loss.
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: type of reduction applied to the output. possible values are
                'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
        """
        super().__init__()
        if (alpha >= 0) and (alpha <= 1):
            self.alpha = alpha
        else:
            warnings.warn(
                f"Depreciated alpha should be between 0 and 1 replaced by \
                    {alpha/(alpha+1.0)}"
            )
            self.alpha = alpha / (alpha + 1.0)
        self.min_margin = min_margin
        self.reduction = reduction
        self.multi_gpu = multi_gpu

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_batch = F.hkr_multiclass_loss(
            input, target, self.alpha, self.min_margin, multi_gpu=self.multi_gpu
        )
        return F.apply_reduction(loss_batch, self.reduction)


class SoftHKRMulticlassLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=0.9,
        min_margin=1.0,
        alpha_mean=0.99,
        temperature=1.0,
        reduction: str = "mean",
    ):
        """
        The multiclass version of HKR with softmax. This is done by computing
        the HKR term over each class and averaging the results.

        [2] M. Serrurier, F. Mamalet, T. Fel et al. "On the explainable properties
        of 1-Lipschitz Neural Networks: An Optimal Transport Perspective.", 2024

        Note that`target` should be one-hot encoded, +/-1 values.

        the regularization factor `alpha` is a value between 0 and 1. It controls the
        trade-off between the hinge and the KR loss. When `alpha` is 0, the loss is
        equivalent to the KR loss, and when `alpha` is 1, the loss is equivalent to the
        hinge loss.

        Args:
            alpha (float): regularization factor (0 <= alpha <= 1),
            min_margin (float): margin to enforce.
            alpha_mean (float): geometric mean factor
            temperature (float): factor for softmax  temperature
                (higher value increases the weight of the highest non y_true logits)
            reduction: type of reduction applied to the output. possible values are
                'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
        """
        if (alpha >= 0) and (alpha <= 1):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            warnings.warn(
                f"Depreciated alpha should be between 0 and 1 replaced by \
                    {alpha/(alpha+1.0)}"
            )
            self.alpha = torch.tensor(alpha / (alpha + 1.0), dtype=torch.float32)
        self.min_margin_v = min_margin
        self.alpha_mean = alpha_mean

        self.current_mean = torch.tensor((self.min_margin_v,), dtype=torch.float32)
        """    constraint=lambda x: torch.clamp(x, 0.005, 1000),
            name="current_mean",
        )"""

        self.temperature = temperature * self.min_margin_v
        if alpha == 1.0:  # alpha = 1.0 => hinge only
            self.fct = self.multiclass_hinge_soft
        else:
            if alpha == 0.0:  # alpha = 0.0 => KR only
                self.fct = self.kr_soft
            else:
                self.fct = self.hkr
        self.reduction = reduction

        super(SoftHKRMulticlassLoss, self).__init__()

    def clamp_current_mean(self, x):
        return torch.clamp(x, 0.005, 1000)

    def _update_mean(self, y_pred):
        self.current_mean = self.current_mean.to(y_pred.device)
        current_global_mean = torch.mean(torch.abs(y_pred)).to(
            dtype=self.current_mean.dtype
        )
        current_global_mean = (
            self.alpha_mean * self.current_mean
            + (1 - self.alpha_mean) * current_global_mean
        )
        self.current_mean = self.clamp_current_mean(current_global_mean).detach()
        total_mean = current_global_mean
        total_mean = torch.clamp(total_mean, self.min_margin_v, 20000)
        return total_mean

    def computeTemperatureSoftMax(self, y_true, y_pred):
        total_mean = self._update_mean(y_pred)
        current_temperature = (
            torch.clamp(self.temperature / total_mean, 0.005, 250)
            .to(dtype=y_pred.dtype)
            .detach()
        )
        min_value = torch.tensor(torch.finfo(torch.float32).min, dtype=y_pred.dtype).to(
            device=y_pred.device
        )
        opposite_values = torch.where(
            y_true > 0, min_value, current_temperature * y_pred
        )
        F_soft_KR = torch.softmax(opposite_values, dim=-1)
        one_value = torch.tensor(1.0, dtype=F_soft_KR.dtype).to(device=y_pred.device)
        F_soft_KR = torch.where(y_true > 0, one_value, F_soft_KR)
        return F_soft_KR

    def signed_y_pred(self, y_true, y_pred):
        """Return for each item sign(y_true)*y_pred."""
        sign_y_true = torch.where(y_true > 0, 1, -1)  # switch to +/-1
        sign_y_true = sign_y_true.to(dtype=y_pred.dtype)
        return y_pred * sign_y_true

    def multiclass_hinge_preproc(self, signed_y_pred, min_margin):
        """From multiclass_hinge(y_true, y_pred, min_margin)
        simplified to use precalculated signed_y_pred"""
        # compute the elementwise hinge term
        hinge = torch.nn.functional.relu(min_margin / 2.0 - signed_y_pred)
        return hinge

    def multiclass_hinge_soft_preproc(self, signed_y_pred, F_soft_KR):
        hinge = self.multiclass_hinge_preproc(signed_y_pred, self.min_margin_v)
        b = hinge * F_soft_KR
        b = torch.sum(b, axis=-1)
        return b

    def multiclass_hinge_soft(self, y_true, y_pred):
        F_soft_KR = self.computeTemperatureSoftMax(y_true, y_pred)
        signed_y_pred = self.signed_y_pred(y_true, y_pred)
        return self.multiclass_hinge_soft_preproc(signed_y_pred, F_soft_KR)

    def kr_soft_preproc(self, signed_y_pred, F_soft_KR):
        kr = -signed_y_pred
        a = kr * F_soft_KR
        a = torch.sum(a, axis=-1)
        return a

    def kr_soft(self, y_true, y_pred):
        F_soft_KR = self.computeTemperatureSoftMax(y_true, y_pred)
        signed_y_pred = self.signed_y_pred(y_true, y_pred)
        return self.kr_soft_preproc(signed_y_pred, F_soft_KR)

    def hkr(self, y_true, y_pred):
        F_soft_KR = self.computeTemperatureSoftMax(y_true, y_pred)
        signed_y_pred = self.signed_y_pred(y_true, y_pred)

        loss_softkr = self.kr_soft_preproc(signed_y_pred, F_soft_KR)

        loss_softhinge = self.multiclass_hinge_soft_preproc(signed_y_pred, F_soft_KR)
        return (1 - self.alpha) * loss_softkr + self.alpha * loss_softhinge

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not (isinstance(input, torch.Tensor)):  # required for dtype.max
            input = torch.Tensor(input, dtype=input.dtype)
        if not (isinstance(target, torch.Tensor)):
            target = torch.Tensor(target, dtype=input.dtype)
        loss_batch = self.fct(target, input)
        return F.apply_reduction(loss_batch, self.reduction)


class LseHKRMulticlassLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        temperature: float = 1.0,
        penalty=1.0,  # max <logsumpexp< max+penalty*margin
        min_margin=1,
        reduction: str = "mean",
    ):
        """
        Loss that estimates the Wasserstein-1 distance using the Kantorovich-Rubinstein
        duality with a hinge regularization and logsumexp summary.
            Args:
                alpha: Regularization factor between the hinge and the KR loss
                        (0 <= alpha <= 1).
                min_margin (float): margin to enforce.
                alpha_mean (float): geometric mean factor
                temperature (float): temperature factor applied on logits
                    (both in KR and Hinge)
                penalty (float): penalty factor for the logsumexp summary
                    max <logsumpexp< max+penalty*margin (default is 1.0).
                min_margin: Minimal margin for the hinge loss.
                reduction: type of reduction applied to the output. possible values are
                   'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
        """
        super().__init__()
        if (alpha >= 0) and (alpha <= 1):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            warnings.warn(
                f"Depreciated alpha should be between 0 and 1 replaced by \
                    {alpha/(alpha+1.0)}"
            )
            self.alpha = torch.tensor(alpha / (alpha + 1.0), dtype=torch.float32)
        self.penalty = penalty
        self.temperature = temperature
        self.margin = min_margin
        if alpha == 1.0:  # alpha = 1.0 => hinge only
            self.fct = self.lse_hinge
        else:
            if alpha == 0.0:  # alpha = 0.0 => KR only
                self.fct = self.lse_kr
            else:
                self.fct = self.lse_hkr
        self.reduction = reduction

    def get_positive(self, y_pred, y_true):
        return y_pred[y_true > 0]

    def compute_lse_neg(self, y_pred, y_true, min_margin):
        neg = torch.where(y_true > 0, -float("inf"), y_pred)
        nb_bins = y_pred.new_tensor(y_pred.size(1) - 1)
        nb_bins = torch.log(nb_bins)
        t = nb_bins / (min_margin * self.penalty / 2.0)  # margin = min_margin/2.0
        lse_neg = (
            1 / t * torch.logsumexp(t * neg, dim=1)
        )  # max <neg_soft< max+penalty*margin
        return lse_neg

    def lse_hinge_preproc(self, pos, lse_neg, min_margin):
        # compute the elementwise hinge term
        hinge_pos = torch.nn.functional.relu(min_margin / 2.0 - pos)
        hinge_neg = torch.nn.functional.relu(min_margin / 2.0 + lse_neg)
        return hinge_pos + hinge_neg

    def lse_hinge(self, y_pred, y_true):
        pos = self.get_positive(y_pred, y_true)
        lse_neg = self.compute_lse_neg(y_pred, y_true, self.margin)
        hinge = self.lse_hinge_preproc(pos, lse_neg, self.margin)
        return hinge

    def lse_kr_preproc(self, pos, lse_neg):
        return pos - lse_neg

    def lse_kr(self, y_pred, y_true):
        pos = self.get_positive(y_pred, y_true)
        lse_neg = self.compute_lse_neg(y_pred, y_true, self.margin)
        kr = -self.lse_kr_preproc(pos, lse_neg)
        return kr

    def lse_hkr(self, y_pred, y_true):
        pos = self.get_positive(y_pred, y_true)
        lse_neg = self.compute_lse_neg(y_pred, y_true, self.margin)

        lse_kr = -self.lse_kr_preproc(pos, lse_neg)
        lse_hinge = self.lse_hinge_preproc(pos, lse_neg, self.margin)
        return (1 - self.alpha) * lse_kr + self.alpha * lse_hinge

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not (isinstance(input, torch.Tensor)):  # required for dtype.max
            input = torch.Tensor(input, dtype=input.dtype)
        if not (isinstance(target, torch.Tensor)):
            target = torch.Tensor(target, dtype=input.dtype)
        input = input * self.temperature
        loss_batch = self.fct(input, target)
        return F.apply_reduction(loss_batch, self.reduction)


class TauCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
        tau: float,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        """
        The loss add a temperature (tau) factor to the CrossEntropyLoss
        CrossEntropyLoss(tau * input, target).

        See `CrossEntropyLoss` for more details on arguments.

        Args:
            tau (float): factor for  temperature
        """

        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.tau = tau

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.shape == target.shape:
            return super().forward(input * self.tau, target.to(torch.double)) / self.tau
        else:
            return super().forward(input * self.tau, target.to(torch.int64)) / self.tau


class TauBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(
        self,
        tau: float,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight=None,
    ) -> None:
        """
        The loss add a temperature (tau) factor to the BCEWithLogitsLoss
        BCEWithLogitsLoss(tau * input, target).

        See `BCEWithLogitsLoss` for more details on arguments.

        Args:
            tau (float): factor for  temperature
        """
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )
        self.tau = tau

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        true_target = torch.where(target > 0, 1.0, 0.0).to(input.dtype)
        return super().forward(input * self.tau, true_target) / self.tau


class CategoricalHingeLoss(torch.nn.Module):
    def __init__(
        self,
        min_margin: float = 1.0,
        reduction: str = "mean",
    ):
        """
        This implementation is sligthly different from the pytorch MultiMarginLoss.

        `target` and `input` must be of shape (batch_size, # classes).
        Note that `target` should be one-hot encoded, +/-1 values.
        :math:`\\text{ReLU}(\\text{min\\_margin} - (\\text{input}[\\text{target}>0]
        - \\text{max}(\\text{input}[\\text{target}<=0])))`
        is computed element-wise and averaged over the batch.

        Args:
            min_margin (float): margin parameter.
            reduction: type of reduction applied to the output. possible values are
                'none' | 'mean' | 'sum' | 'auto'; default is 'mean' ('auto is 'mean')
        """
        super().__init__()
        self.min_margin = min_margin
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = torch.where(target > 0, 1, 0).to(input.dtype)
        pos = torch.sum(mask * input, dim=-1)
        neg = torch.max(
            torch.where(target > 0, input - self.min_margin, input), dim=-1
        ).values
        loss_batch = torch.nn.functional.relu(self.min_margin - (pos - neg))
        return F.apply_reduction(loss_batch, self.reduction)
