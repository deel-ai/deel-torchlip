from typing import Optional, List
import torch
import torch.nn as nn
import torch.distributed as dist

from .module import ScaledLipschitzModule, SharedLipFactory


class LayerCentering(nn.Module):
    r"""
    Applies Layer centering over a mini-batch of inputs.

    This layer implements the operation as described in
    .. math::
        y = x - \mathrm{E}[x] + \beta
    The mean is calculated over the last `D` dimensions
    given in the `dim` parameter.
    `\beta` is learnable bias parameter. that can be
    applied after the mean subtraction.
    Unlike Layer Normalization, this layer is 1-Lipschitz
    This layer uses statistics computed from input data in
    both training and  evaluation modes.
    This layer is :math:`1`-Lipschitz and should be used

    Args:
        num_features: number of features in the input tensor
        dim: dimensions over which to compute the mean
        (default ``input.mean((-2, -1))`` for a 4D tensor).
        bias: if `True`, adds a learnable bias to the output
        of shape (num_features,). Default: `True`

    Shape:
        - Input: :math:`(N, num_features, *)`
        - Output: :math:`(N, num_features, *)` (same shape as input)

    """

    def __init__(self, num_features: int = 1, dim: tuple = [-2, -1], bias=True):
        super(LayerCentering, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_features,)), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        if self.bias is not None:
            bias_shape = (1, -1) + (1,) * (len(x.shape) - 2)
            return x - mean + self.bias.view(bias_shape)
        else:
            return x - mean


LayerCentering2d = LayerCentering


class ScaleBiasLayer(nn.Module):
    def __init__(
        self,
        scalar=1.0,
        num_features: int = 1,
        bias: bool = True,
    ):
        """
        A PyTorch layer that multiplies the input by a fixed scalar.
        and add a bias. This module is used to export
        Lipschitz normalization layers into vanilla layers.
        Args:
            scalar: The scalar multiplier (non-learnable).
            num_features: number of features in the input tensor
            bias: if `True`, adds a learnable bias to the output
            of shape (size,). Default: `True`
        """
        super(ScaleBiasLayer, self).__init__()
        self.scalar = scalar
        self.num_features = num_features
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_features,)), requires_grad=True)
            self.bias_shape = None

    def forward(self, x):
        if self.bias is not None:
            if self.bias_shape is None:
                self.bias_shape = (1, -1) + (1,) * (len(x.shape) - 2)
            return x * self.scalar + self.bias.view(self.bias_shape)
        else:
            return x * self.scalar


class BatchLipNorm(nn.Module, ScaledLipschitzModule):
    r"""
    Applies Batch Normalization with a single scaling factor over a 2D, 3D, 4D input.

    .. math::

        y_i = (x_i - \mathrm{E}[x_i])/\sqrt{\max{\mathrm{Var}[x_i]+\epsilon}} + \beta_i

    The mean is calculated per-dimension over the mini-batches and
    other dimensions excepted the feature/channel dimension.
        :math:`\beta` is a learnable parameter vectors
    of num_features `C` (where `C` is the number of features or channels of the input).
    This layer uses statistics computed from input data in
    training mode and  a constant in evaluation mode computed as
    the running mean on training samples.
        This layer is compatible with multi-GPU training (torch.nn.distributed).
    Warning: This layer is not :math:`1`-Lipschitz and could be used with a
    SharedLipFactory to compute the global Lipschitz constant on Sequential models.

    Args:
            num_features: number of features in the input tensor
            dim: dimensions over which to compute the mean
        (default ``input.mean((0, -2, -1))`` for a 4D tensor).
            centering: apply a batch centering (\mathrm{E}[x_i]) if `True`
        otherwise only apply the variance normalization.
            bias: if `True`, adds a learnable bias to the output
        of shape (num_features,). Default: `True`
            normalize: apply variance normalization if `True`
        otherwise only apply centering. Note that BatchCentering class
        is equivalent to BatchLipNorm with `normalize=False`.
        By default `normalize=True`.
            factory (optional): SharedLipFactory to register the scaling factor
            eps: a value added to the denominator for numerical stability.

    Shape:
        - Input: :math:`(N, num_features, *)`
        - Output: :math:`(N, num_features, *)` (same shape as input)

    """

    def __init__(
        self,
        num_features: int = 1,
        dim: Optional[tuple] = None,
        centering: bool = True,
        bias: bool = True,
        normalize: bool = True,
        factory: Optional[SharedLipFactory] = None,
        eps: float = 1e-5,
    ):
        nn.Module.__init__(self)
        ScaledLipschitzModule.__init__(self, scaling=normalize, factory=factory)
        self.num_features = num_features
        self.dim = dim
        self.centering = centering
        self.eps = eps

        # register for saving the local mean on batch
        # running_sum_bmean sum of batch mean (mean needs it)
        self.register_buffer("running_sum_bmean", torch.zeros((num_features,)))
        self.register_buffer("running_num_batches", torch.zeros((1,), dtype=torch.long))
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_features,)), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        self.normalize = normalize
        if self.normalize:
            # register running_sum_bmeansq for saving the sum of mean of square on batch
            self.register_buffer("running_sum_bmeansq", torch.zeros(num_features))

            # registers for storing the total number samples in the epoch
            # Need two buffers to keep the total number when averaging
            # at the end of the epoch
            # the epoch average values will be considered as representing one batch
            #  only for the next epoch
            self.register_buffer("running_mean_sample_per_batches", torch.zeros((1,)))
            self.register_buffer("total_num_samples", torch.zeros((1,)))

        # Cached batch stats (for training-time scaling and mean)
        # these cached values are kept outside the forward function
        # to get access in the Lipschitz computation
        self._batch_mean: Optional[torch.Tensor] = None
        self._batch_meansq: Optional[torch.Tensor] = None
        self._batch_count: Optional[torch.Tensor] = None
        self.local_num_elements: Optional[int] = None

        self._first = True

    def _infer_dim(self, x: torch.Tensor) -> tuple:
        if self.dim is None:
            # Default: reduce over batch + spatial dims, keep channel
            self.dim = (0,) + tuple(range(2, x.dim()))

    @staticmethod
    def _all_reduce_sum_(lt: List[torch.Tensor]):
        if dist.is_available() and dist.is_initialized():
            for t in lt:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

    def reset_states(self):
        self.running_sum_bmean.zero_()
        self.running_num_batches.zero_()
        if self.normalize:
            self.running_sum_bmeansq.zero_()
            self.running_mean_sample_per_batches.zero_()
            self.total_num_samples.zero_()

    # compute average of running values
    def update_running_values(self):
        """
        In eval mode reduce the running values to their mean
        and set the counter of batches to one
        (previous epoch will have a weight equivalent to one batch)
        """
        if self.running_num_batches > 1:
            self._batch_mean = None  # free some memory
            self._batch_meansq = None
            self._batch_count = None
            self.running_sum_bmean = self.running_sum_bmean / self.running_num_batches
            if self.normalize:
                self.running_sum_bmeansq = (
                    self.running_sum_bmeansq / self.running_num_batches
                )
                self.total_num_samples = self.running_mean_sample_per_batches
                self.running_mean_sample_per_batches = (
                    self.running_mean_sample_per_batches / self.running_num_batches
                )
            self.running_num_batches = self.running_num_batches.zero_() + 1

    def _get_mean_and_update(self, training: bool = False) -> torch.Tensor:
        """retrieve the batch mean in training mode
        update and retrieve the running mean in eval mode
        """
        assert training is False, "use _get_mean only in training mode"
        self.update_running_values()
        return self._get_mean(training=False)

    def _get_mean(self, training: bool = False) -> torch.Tensor:
        """retrieve the batch mean in training mode
        and the running mean in eval mode"""

        if training:
            return self._batch_mean

        # eval mode: return running mean
        # case asking for running mean before a step
        if self.running_num_batches.item() == 0:
            return torch.zeros_like(self.running_sum_bmean)
        # self.running_num_batches is set to 1 after update_running_values
        return self.running_sum_bmean / self.running_num_batches

    def _get_var(self, training: bool = False) -> torch.Tensor:
        """retrieve the batch variance in training mode
        and the running variance in eval mode,
        when variance is too small return 1.0 to avoid division by zero
        """
        cur_mean = self._get_mean(training)
        if cur_mean is None:
            raise RuntimeError(
                f"Mean is None when computing variance. training={training}"
            )
        if not self.normalize:
            return torch.ones_like(cur_mean)
        if training:
            # rectified local variance using local buffers:
            # var_sum is sum_on_batch(x_i^2)/current_sample_num
            # where current_sample_num batch size
            # variance =
            # (var_sum - (cur_mean*cur_mean))*current_sample_num/(current_sample_num-1)
            cur_meansq = self._batch_meansq
            cur_sample_num = self.local_num_elements
            assert (
                cur_sample_num is not None
            ), "current_sample_num should be provided in training mode"
            var_factor = cur_sample_num / (cur_sample_num - 1)
            # num_batches = 1.0
        else:
            if self.running_num_batches.item() == 0:
                return torch.ones_like(self.running_sum_bmeansq)
            # rectified variance using running buffers:
            # var_sum is sum_on_epoch(x_i^2)/total_num_samples where
            # total_num_samples number of samples
            # variance =
            # (var_sum - (cur_mean*cur_mean))*total_num_samples/(total_num_samples-1)
            cur_meansq = self.running_sum_bmeansq / self.running_num_batches
            if self.total_num_samples.item() == 0:  # in case of update = False
                self.total_num_samples = self.running_mean_sample_per_batches
            total_num_samples = self.total_num_samples
            var_factor = total_num_samples / (total_num_samples - 1)
            # num_batches = self.running_num_batches

        var = (cur_meansq - (cur_mean * cur_mean)) * var_factor
        var = torch.where(var < self.eps, torch.ones(var.shape).to(var.device), var)
        return var

    def get_scaling(self, training: bool = False):
        """retrieve the scaling factor based on the maximum of the variance
        this is the main difference with batchnorm that uses per-feature variance
        """
        var = self._get_var(training)
        max_var = var.max()
        return 1.0 / max_var.sqrt()

    def forward(self, x):
        self._infer_dim(x)
        mean_shape = (1, -1) + (1,) * (len(x.shape) - 2)
        self.local_num_elements = x.numel() // x.size(1)  # x[:, 0].numel()

        if self.training:
            # on first batch initalize variables
            if self._first:
                self.reset_states()
                self._first = False
            # compute local mean (on batch of a single GPU)
            if self.centering:
                self._batch_mean = x.mean(dim=self.dim)
            else:
                self._batch_mean = torch.zeros((self.num_features,)).to(x.device)

            # compute local mean square (on batch of a single GPU)
            if self.normalize:
                xsq = x * x
                self._batch_meansq = xsq.mean(dim=self.dim)

            aggregated_mean = self._batch_mean.clone().detach()
            # on a single GPU this value is always 1
            num_batches = torch.zeros_like(self.running_num_batches) + 1
            # for multiGPU aggregate mean and num_batches
            list_tensors = [aggregated_mean, num_batches]
            if self.normalize:
                current_meansq = self._batch_meansq.clone()
                list_tensors.append(current_meansq.detach())
            self._all_reduce_sum_(list_tensors)

            # Accumulate running mean, mean square and num elements over the epoch
            # use aggregated mean and mean square for multi GPU
            with torch.no_grad():
                self.running_sum_bmean += aggregated_mean
                self.running_num_batches += num_batches
                if self.normalize:
                    self.running_sum_bmeansq += current_meansq
                    self.running_mean_sample_per_batches += (
                        self.local_num_elements * num_batches
                    )
            mean = self._get_mean(self.training)
        else:
            mean = self._get_mean_and_update(training=self.training)

        # get scaling factor: shape = (C,)
        # get scaling factor: shape = (1,)
        scale = self.get_scaling(training=self.training).to(x.device)

        if self.bias is not None:
            return (x - mean.view(mean_shape)) * scale + self.bias.view(mean_shape)
        else:
            return (x - mean.view(mean_shape)) * scale

    def vanilla_export(self):
        num_features = self.running_sum_bmean.shape[0]
        bias = -self._get_mean_and_update(training=False).detach()

        scalef = 1.0
        if self.normalize:
            scalef = self.get_scaling(training=False).detach()
            bias *= scalef
        # self.running_sum_bmean.detach() / self.running_num_batches.detach()
        if self.bias is not None:
            bias += self.bias.detach()

        layer = ScaleBiasLayer(scalar=scalef, bias=True, num_features=num_features)
        layer.bias.data = bias
        return layer


class BatchCentering(BatchLipNorm):
    r"""
    Applies Batch Centering  over a 2D, 3D, 4D input.

    .. math::

        y_i = (x_i - \mathrm{E}[x_i]) + \beta_i
    BatchCentering implemented as BatchLipNorm with normalize = False,
    factory=None and centering=True.

    The mean is calculated per-dimension over the mini-batches and
    other dimensions excepted the feature/channel dimension.
        :math:`\beta` is a learnable parameter vectors
    of num_features `C` (where `C` is the number of features or channels of the input).
    This layer uses statistics computed from input data in
    training mode and  a constant in evaluation mode computed as
    the running mean on training samples.
        This layer is compatible with multi-GPU training (torch.nn.distributed).
    This layer is :math:`1`-Lipschitz and should be used

    Args:
        num_features: number of features in the input tensor
        dim: dimensions over which to compute the mean
        (default ``input.mean((0, -2, -1))`` for a 4D tensor).
        bias: if `True`, adds a learnable bias to the output
        of shape (num_features,). Default: `True`

    Shape:
        - Input: :math:`(N, num_features, *)`
        - Output: :math:`(N, num_features, *)` (same shape as input)

    """

    def __init__(
        self,
        num_features: int = 1,
        dim: Optional[tuple] = None,
        bias: bool = True,
    ):
        super().__init__(
            num_features=num_features,
            dim=dim,
            centering=True,  # Batchcentering is centering
            bias=bias,
            normalize=False,  # forces scaling_norm = 1
            factory=None,  # forces scaling_norm = 1
        )


BatchCentering2d = BatchCentering
