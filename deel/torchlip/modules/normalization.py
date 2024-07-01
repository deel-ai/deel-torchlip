import torch
import torch.nn as nn
import torch.distributed as dist


class LayerCentering(nn.Module):
    def __init__(self, size=-1, dim=[-2, -1], bias=True):
        super(LayerCentering, self).__init__()
        self.bias = bias
        if isinstance(size, tuple):
            self.alpha = nn.Parameter(torch.zeros(size), requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.zeros(1, size, 1, 1), requires_grad=True)
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        if self.bias:
            return x - mean + self.alpha
        return x - mean


class LayerCentering2D(LayerCentering):
    def __init__(self, size=1, dim=[-2, -1]):
        super(LayerCentering2D, self).__init__(size=size, dim=[-2, -1])


class BatchCentering(nn.Module):
    def __init__(self, size=1, dim=[0, -2, -1], momentum=0.05):
        super(BatchCentering, self).__init__()
        self.dim = dim
        self.momentum = momentum
        if isinstance(size, tuple):
            self.register_buffer("running_mean", torch.zeros(size))
        else:
            self.register_buffer("running_mean", torch.zeros(1, size, 1, 1))

        self.first = True

    def forward(self, x):

        if self.training:
            mean = x.mean(dim=self.dim, keepdim=True)
            # print(mean.shape)
            with torch.no_grad():
                if self.first:
                    # print("first")
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
        return x - mean


class BatchCenteringBiases(BatchCentering):
    def __init__(self, size=1, dim=[0, -2, -1], momentum=0.05):
        super(BatchCenteringBiases, self).__init__(
            size=size, dim=dim, momentum=momentum
        )
        if isinstance(size, tuple):
            self.alpha = nn.Parameter(torch.zeros(size), requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.zeros(1, size, 1, 1), requires_grad=True)

    def forward(self, x):
        return super().forward(x) + self.alpha


class BatchCenteringBiases2D(BatchCenteringBiases):
    def __init__(self, size=1, momentum=0.05):
        super(BatchCenteringBiases2D, self).__init__(
            size=size, dim=[0, -2, -1], momentum=momentum
        )


class BatchCentering2D(BatchCentering):
    def __init__(self, size=1, momentum=0.05):
        super(BatchCentering2D, self).__init__(
            size=size, dim=[0, -2, -1], momentum=momentum
        )
