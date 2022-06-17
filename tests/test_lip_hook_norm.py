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

from deel.torchlip.normalizers import bjorck_normalization
from deel.torchlip.utils import bjorck_norm
from deel.torchlip.utils import frobenius_norm
from deel.torchlip.utils import lconv_norm
from deel.torchlip.utils import remove_bjorck_norm
from deel.torchlip.utils import remove_frobenius_norm
from deel.torchlip.utils import remove_lconv_norm
from deel.torchlip.utils.lconv_norm import compute_lconv_coef


def test_bjorck_norm():
    """
    test bjorck_norm hook implementation
    """
    m = torch.nn.Linear(2, 2)
    torch.nn.init.orthogonal_(m.weight)
    w1 = bjorck_normalization(m.weight)

    bjorck_norm(m)
    assert not isinstance(m.weight, torch.nn.Parameter)

    x = torch.rand(2)
    m(x)
    tt.assert_equal(w1, m.weight)

    remove_bjorck_norm(m)
    assert isinstance(m.weight, torch.nn.Parameter)
    tt.assert_equal(w1, m.weight)


def test_frobenius_norm():
    """
    test frobenius_norm hook implementation
    """
    m = torch.nn.Linear(2, 2)
    torch.nn.init.uniform_(m.weight)
    w1 = m.weight / torch.norm(m.weight)

    frobenius_norm(m, disjoint_neurons=False)
    assert not isinstance(m.weight, torch.nn.Parameter)

    x = torch.rand(2)
    m(x)
    tt.assert_equal(w1, m.weight)

    remove_frobenius_norm(m)
    assert isinstance(m.weight, torch.nn.Parameter)
    tt.assert_equal(w1, m.weight)


def test_frobenius_norm_disjoint_neurons():
    """
    Test `disjoint_neurons=True` parameter in frobenius_norm hook
    """
    m = torch.nn.Linear(in_features=5, out_features=3)

    # Set hook and perform a forward pass to compute new weights
    frobenius_norm(m, disjoint_neurons=True)
    m(torch.rand(5))

    # Assert that all rows of matrix weight are independently normalized
    for i in range(m.out_features):
        tt.assert_allclose(torch.norm(m.weight[i, :]), torch.tensor(1.0), rtol=2e-7)


def test_lconv_norm():
    """
    test lconv_norm hook implementation
    """
    m = torch.nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
    torch.nn.init.orthogonal_(m.weight)
    w1 = m.weight * compute_lconv_coef(m.kernel_size, (1, 1, 5, 5), m.stride)

    lconv_norm(m)
    assert not isinstance(m.weight, torch.nn.Parameter)

    x = torch.rand(1, 1, 5, 5)
    m(x)
    tt.assert_equal(w1, m.weight)

    remove_lconv_norm(m)
    assert isinstance(m.weight, torch.nn.Parameter)
    tt.assert_equal(w1, m.weight)
