from typing import Optional
import torch
import torch.nn as nn
import torch.distributed as dist


class LayerCentering(nn.Module):
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
