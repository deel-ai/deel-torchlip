# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

from typing import Tuple

import torch


class InvertibleDownSampling(torch.nn.Module):
    def __init__(
        self,
        pool_size: Tuple[int, int],
    ):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            tuple(
                input[:, :, i :: self.pool_size[0], j :: self.pool_size[1]]
                for i in range(self.pool_size[0])
                for j in range(self.pool_size[1])
            ),
            dim=1,
        )
