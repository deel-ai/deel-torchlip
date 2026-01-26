from typing import List, Optional
import torch
import torch.nn as nn
import torch.distributed as dist


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

    Args:
        size: number of features in the input tensor
        dim: dimensions over which to compute the mean
        (default ``input.mean((-2, -1))`` for a 4D tensor).
        bias: if `True`, adds a learnable bias to the output
        of shape (size,). Default: `True`

    Shape:
        - Input: :math:`(N, size, *)`
        - Output: :math:`(N, size, *)` (same shape as input)

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
        and add a bias
        :param scalar: The scalar multiplier (non-learnable).
        :param size: number of features in the input tensor
        :param bias: if `True`, adds a learnable bias to the output
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


class BatchLipNorm(nn.Module):
    r"""
    Applies Batch Centering  over a 2D, 3D, 4D input.

    .. math::

        y_i = x_i - \mathrm{E}[x_i] + \beta_i

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
        centering: bool = True,
        bias: bool = True,
        factory=None,
    ):
        super(BatchLipNorm, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.centering = centering

        # register for saving the local mean on batch
        # running_sum_bmean sum of batch mean (mean needs it)
        self.register_buffer("running_sum_bmean", torch.zeros((num_features,)))
        self.register_buffer("running_num_batches", torch.zeros((1,), dtype=torch.long))
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_features,)), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        # Cached batch stats (for training-time scaling and mean)
        self._batch_mean: Optional[torch.Tensor] = None

        self._first = True
        self.factory = factory

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

    # compute average of running values
    def update_running_values(self):
        if self.running_num_batches > 1:
            self.running_sum_bmean = self.running_sum_bmean / self.running_num_batches
            self.running_num_batches = self.running_num_batches.zero_() + 1

    def _get_mean(self, training: bool, update=True) -> torch.Tensor:
        """Retrieve the running mean in eval mode."""
        """ If update is True, update the running values"""
        if training:
            return self._batch_mean

        # case asking for running mean before a step
        if self.running_num_batches.item() == 0:
            return torch.zeros_like(self.running_sum_bmean)
        if update and (self.running_num_batches > 1):
            self.update_running_values()
        return self.running_sum_bmean / self.running_num_batches

    def forward(self, x):
        self._infer_dim(x)
        mean_shape = (1, -1) + (1,) * (len(x.shape) - 2)
        if self.training:
            if self._first:
                self.reset_states()
                self._first = False
            # compute local mean (on batch of a single GPU)
            if self.centering:
                self._batch_mean = x.mean(dim=self.dim)
            else:
                self._batch_mean = torch.zeros((self.num_features,)).to(x.device)
            agrregated_mean = self._batch_mean.clone().detach()
            # on a single GPU this value is always 1
            num_batches = self.running_num_batches.clone().detach().zero_() + 1
            # for multiGPU aggregate mean and num_batches
            list_tensors = [agrregated_mean, num_batches]
            self._all_reduce_sum_(list_tensors)

            with torch.no_grad():
                self.running_sum_bmean += agrregated_mean
                self.running_num_batches += num_batches

        mean = self._get_mean(self.training)

        if self.bias is not None:
            return x - mean.view(mean_shape) + self.bias.view(mean_shape)
        else:
            return x - mean.view(mean_shape)

    def vanilla_export(self):
        num_features = self.running_sum_bmean.shape[0]
        bias = -self.running_sum_bmean.detach() / self.running_num_batches.detach()
        if self.bias is not None:
            bias += self.bias.detach()

        layer = ScaleBiasLayer(scalar=1.0, bias=True, num_features=num_features)
        layer.bias.data = bias
        return layer


class BatchCentering(BatchLipNorm):
    """
    BatchCentering implemented as BatchLipNorm with factory=None and centering=True.
    Equivalent forward: y = x - E[x] + beta (if bias=True).
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
            factory=None,  # forces scaling_norm = 1
        )


BatchCentering2d = BatchCentering
