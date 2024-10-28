from typing import Optional
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

    def __init__(self, size: int = 1, dim: tuple = [-2, -1], bias=True):
        super(LayerCentering, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros((size,)), requires_grad=True)
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


class BatchCentering(nn.Module):
    r"""
    Applies Batch Normalization over a 2D, 3D, 4D input.

    .. math::

        y = x - \mathrm{E}[x] + \beta

    The mean is calculated per-dimension over the mini-batchesa and
    other dimensions excepted the feature/channel dimension.
    This layer uses statistics computed from input data in
    training mode and  a constant in evaluation mode computed as
    the running mean on training samples.
    :math:`\beta` is a learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input).
    that can be applied after the mean subtraction.
    Unlike Batch Normalization, this layer is 1-Lipschitz

    Args:
        size: number of features in the input tensor
        dim: dimensions over which to compute the mean
        (default ``input.mean((0, -2, -1))`` for a 4D tensor).
        momentum: the value used for the running mean computation
        bias: if `True`, adds a learnable bias to the output
        of shape (size,). Default: `True`

    Shape:
        - Input: :math:`(N, size, *)`
        - Output: :math:`(N, size, *)` (same shape as input)

    """

    def __init__(
        self,
        size: int = 1,
        dim: Optional[tuple] = None,
        momentum: float = 0.05,
        bias: bool = True,
    ):
        super(BatchCentering, self).__init__()
        self.dim = dim
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros((size,)))
        if bias:
            self.bias = nn.Parameter(torch.zeros((size,)), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        self.first = True

    def forward(self, x):
        if self.dim is None:  # (0,2,3) for 4D tensor; (0,) for 2D tensor
            self.dim = (0,) + tuple(range(2, len(x.shape)))
        mean_shape = (1, -1) + (1,) * (len(x.shape) - 2)
        if self.training:
            mean = x.mean(dim=self.dim)
            with torch.no_grad():
                if self.first:
                    self.running_mean = mean
                    self.first = False
                else:
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * mean
            if dist.is_initialized():
                dist.all_reduce(self.running_mean, op=dist.ReduceOp.SUM)
                self.running_mean /= dist.get_world_size()
        else:
            mean = self.running_mean
        if self.bias is not None:
            return x - mean.view(mean_shape) + self.bias.view(mean_shape)
        else:
            return x - mean.view(mean_shape)


BatchCentering2d = BatchCentering
