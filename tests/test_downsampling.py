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
import torch
import torch_testing as tt

from deel.torchlip.functional import invertible_downsample


def test_invertible_downsample():

    # 1D input
    x = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    x = invertible_downsample(x, (2,))
    assert x.shape == (1, 4, 2)

    # TODO: Check this.
    tt.assert_equal(x, torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]]))

    # 2D input
    x = torch.rand(10, 1, 128, 128)
    assert invertible_downsample(x, (4, 4)).shape == (10, 16, 32, 32)

    x = torch.rand(10, 4, 64, 64)
    assert invertible_downsample(x, (2, 2)).shape == (10, 16, 32, 32)

    # 3D input
    x = torch.rand(10, 2, 128, 64, 64)
    assert invertible_downsample(x, 2).shape == (10, 16, 64, 32, 32)
