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
import os
import pytest

import numpy as np

from . import utils_framework as uft

from .utils_framework import (
    BatchCentering,
    BatchLipNorm,
    LayerCentering,
    SharedLipFactory,
)


def check_serialization(
    layer_type, layer_params, input_shape=(10,), norm_factory=False
):
    if norm_factory:
        factory = SharedLipFactory()
        layer_params["factory"] = factory
    m = uft.generate_k_lip_model(layer_type, layer_params, input_shape=input_shape, k=1)
    if m is None:
        pytest.skip()
    loss, optimizer, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=uft.CategoricalCrossentropy(from_logits=True),
    )
    name = layer_type.__class__.__name__
    path = os.path.join("logs", "normalization", name)
    xnp = np.random.uniform(-10, 10, (255,) + input_shape)
    x = uft.to_tensor(xnp)
    y1 = m(x)
    uft.save_model(m, path)
    m2 = uft.load_model(
        path,
        compile=True,
        layer_type=layer_type,
        layer_params=layer_params,
        input_shape=input_shape,
        k=1,
    )
    y2 = m2(x)
    np.testing.assert_allclose(uft.to_numpy(y1), uft.to_numpy(y2))


@pytest.mark.skipif(
    hasattr(LayerCentering, "unavailable_class"),
    reason="LayerCentering not available",
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (4, (3, 4, 8, 8), False),
        (4, (3, 4, 8, 8), True),
    ],
)
def test_LayerCentering(size, input_shape, bias):
    """evaluate layerbatch centering"""
    input_shape = uft.to_framework_channel(input_shape)
    x = np.arange(np.prod(input_shape)).reshape(input_shape)
    bn = uft.get_instance_framework(
        LayerCentering, {"num_features": size, "bias": bias}
    )

    mean_x = np.mean(x, axis=(2, 3))
    mean_shape = (-1, size, 1, 1)
    x = uft.to_tensor(x)
    y = bn(x)
    np.testing.assert_allclose(
        uft.to_numpy(y), x - np.reshape(mean_x, mean_shape), atol=1e-5
    )
    y = bn(2 * x)
    np.testing.assert_allclose(
        uft.to_numpy(y), 2 * x - 2 * np.reshape(mean_x, mean_shape), atol=1e-5
    )  # keep substract batch mean
    bn.eval()
    y = bn(2 * x)
    np.testing.assert_allclose(
        uft.to_numpy(y), 2 * x - 2 * np.reshape(mean_x, mean_shape), atol=1e-5
    )  # eval mode use running_mean


@pytest.mark.skipif(
    hasattr(BatchCentering, "unavailable_class"),
    reason="BatchCentering not available",
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (4, (3, 4), False),
        (4, (3, 4), True),
        (4, (3, 4, 8, 8), False),
        (4, (3, 4, 8, 8), True),
    ],
)
def test_BatchCentering(size, input_shape, bias):
    """evaluate layerbatch centering"""
    input_shape = uft.to_framework_channel(input_shape)
    x = np.arange(np.prod(input_shape)).reshape(input_shape)
    bn = uft.get_instance_framework(
        BatchCentering, {"num_features": size, "bias": bias}
    )
    if len(input_shape) == 2:
        mean_x = np.mean(x, axis=0)
        mean_shape = (1, size)
    else:
        mean_x = np.mean(x, axis=(0, 2, 3))
        mean_shape = (1, size, 1, 1)
    x = uft.to_tensor(x)
    y = bn(x)
    np.testing.assert_allclose(bn.running_sum_bmean, mean_x, atol=1e-5)
    np.testing.assert_allclose(
        uft.to_numpy(y), x - np.reshape(mean_x, mean_shape), atol=1e-5
    )
    y = bn(2 * x)
    new_runningmean = (mean_x + 2 * mean_x) / 2.0
    np.testing.assert_allclose(
        bn.running_sum_bmean / bn.running_num_batches, new_runningmean, atol=1e-5
    )
    np.testing.assert_allclose(
        uft.to_numpy(y), 2 * x - 2 * np.reshape(mean_x, mean_shape), atol=1e-5
    )  # keep substract batch mean
    bn.eval()
    y = bn(2 * x)
    np.testing.assert_allclose(
        bn.running_sum_bmean, new_runningmean, atol=1e-5
    )  # eval mode running mean freezed
    np.testing.assert_allclose(
        uft.to_numpy(y), 2 * x - np.reshape(new_runningmean, mean_shape), atol=1e-5
    )  # eval mode use running_mean


@pytest.mark.parametrize(
    "norm_type",
    [LayerCentering, BatchCentering],
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (10, (10,), False),
        (10, (10,), True),
        (7, (7, 8, 8), False),
        (7, (7, 8, 8), True),
    ],
)
def test_Normalization_serialization(norm_type, size, input_shape, bias):
    # Check serialization
    if hasattr(norm_type, "unavailable_class"):
        pytest.skip(f"{norm_type} not available")
    check_serialization(
        norm_type,
        layer_params={"num_features": size, "bias": bias},
        input_shape=input_shape,
    )


def linear_generator(batch_size, input_shape: tuple):
    """
    Generate data according to a linear kernel
    Args:
        batch_size: size of each batch
        input_shape: shape of the desired input

    Returns:
        a generator for the data

    """
    input_shape = tuple(input_shape)
    while True:
        # pick random sample in [0, 1] with the input shape
        batch_x = np.array(
            np.random.uniform(-10, 10, (batch_size,) + input_shape), dtype=np.float16
        )
        # apply the k lip linear transformation
        batch_y = batch_x
        yield batch_x, batch_y


@pytest.mark.parametrize(
    "norm_type",
    [LayerCentering, BatchCentering],
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (10, (10,), True),
        (7, (7, 8, 8), True),
    ],
)
def test_Normalization_bias(norm_type, size, input_shape, bias):
    if hasattr(norm_type, "unavailable_class"):
        pytest.skip(f"{norm_type} not available")
    m = uft.generate_k_lip_model(
        norm_type,
        layer_params={"num_features": size, "bias": bias},
        input_shape=input_shape,
        k=1,
    )
    if m is None:
        pytest.skip()
    loss, optimizer, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=uft.CategoricalCrossentropy(from_logits=True),
    )
    batch_size = 10
    bb = uft.to_numpy(uft.get_layer_by_index(m, 0).bias)
    np.testing.assert_allclose(bb, np.zeros((size,)), atol=1e-5)

    traind_ds = linear_generator(batch_size, input_shape)
    uft.train(
        traind_ds,
        m,
        loss,
        optimizer,
        2,
        batch_size,
        steps_per_epoch=10,
    )

    bb = uft.to_numpy(uft.get_layer_by_index(m, 0).bias)
    assert np.linalg.norm(bb) != 0.0


@pytest.mark.skipif(
    hasattr(BatchCentering, "unavailable_class"),
    reason="BatchCentering not available",
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (4, (3, 4), False),
        (4, (3, 4), True),
        (4, (3, 4, 8, 8), False),
        (4, (3, 4, 8, 8), True),
    ],
)
def test_BatchCentering_runningmean(size, input_shape, bias):
    """evaluate batch centering convergence of running mean"""
    input_shape = uft.to_framework_channel(input_shape)
    # start with 0 to set up running mean to zero
    x = np.zeros(input_shape)
    bn = uft.get_instance_framework(
        BatchCentering, {"num_features": size, "bias": bias}
    )
    x = uft.to_tensor(x)
    y = bn(x)

    np.testing.assert_allclose(bn.running_sum_bmean, 0.0, atol=1e-5)

    x = np.random.normal(0.0, 1.0, input_shape)
    if len(input_shape) == 2:
        mean_x = np.mean(x, axis=0)
    else:
        mean_x = np.mean(x, axis=(0, 2, 3))
    x = uft.to_tensor(x)
    for _ in range(1000):
        y = bn(x)  # noqa: F841

    np.testing.assert_allclose(
        bn.running_sum_bmean / bn.running_num_batches, mean_x * 1000 / 1001, atol=1e-4
    )
    bn.eval()
    y = bn(x)
    # updated running mean used in eval mode
    np.testing.assert_allclose(bn.running_sum_bmean, mean_x * 1000 / 1001, atol=1e-4)
    x_shape = uft.to_numpy(x).shape
    meanr = np.reshape(mean_x * 1000 / 1001, (1, -1) + (1,) * (len(x_shape) - 2))
    np.testing.assert_allclose(uft.to_numpy(y), uft.to_numpy(x) - meanr, atol=1e-4)


@pytest.mark.parametrize(
    "layer_type,layer_params",
    [
        (BatchCentering, {"num_features": 10, "bias": False}),
        (BatchCentering, {"num_features": 10, "bias": True}),
        (BatchLipNorm, {"num_features": 10, "centering": True, "bias": False}),
        (BatchLipNorm, {"num_features": 10, "centering": True, "bias": True}),
        (BatchLipNorm, {"num_features": 10, "centering": False, "bias": False}),
        (BatchLipNorm, {"num_features": 10, "centering": False, "bias": True}),
    ],
)
@pytest.mark.parametrize("input_shape", [(10, 10), (), (10,)])
@pytest.mark.parametrize(
    "norm",
    [False, True],
)
@pytest.mark.parametrize(
    "norm_factory",
    [False, True],
)
def test_batchcenter_vanilla_export(
    layer_type, layer_params, input_shape, norm, norm_factory
):
    input_shape = (layer_params["num_features"],) + input_shape
    input_shape = uft.to_framework_channel(input_shape)
    if layer_type == BatchLipNorm:
        layer_params["normalize"] = norm
        if norm and norm_factory:
            factory = SharedLipFactory()
            layer_params["factory"] = factory
    model = uft.generate_k_lip_model(layer_type, layer_params, input_shape, 1.0)

    x = np.random.normal(size=(5,) + input_shape)

    x = uft.to_tensor(x)
    y1 = model(x)
    for _ in range(1000):
        y = model(x)  # noqa: F841

    model.eval()
    y1 = model(x)
    # Test vanilla export inference comparison
    if uft.vanilla_require_a_copy():
        model2 = uft.generate_k_lip_model(layer_type, layer_params, input_shape, 1.0)
        uft.copy_model_parameters(model, model2)
        vanilla_model = uft.vanillaModel(model2)
    else:
        vanilla_model = uft.vanillaModel(model)  # .vanilla_export()
    y2 = vanilla_model(x)
    assert type(list(vanilla_model.children())[0]) is uft.ScaleBiasLayer
    np.testing.assert_allclose(uft.to_numpy(y1), uft.to_numpy(y2), atol=1e-6)


@pytest.mark.parametrize(
    "size, input_shape, bias, norm",
    [
        (4, (3, 4), False, False),
        (4, (3, 4), False, True),
        (4, (3, 4), True, False),
        (4, (3, 4), True, True),
        (4, (3, 4, 8, 8), False, False),
        (4, (3, 4, 8, 8), False, True),
        (4, (3, 4, 8, 8), True, False),
        (4, (3, 4, 8, 8), True, True),
    ],
)
def test_BatchLipNorm(size, input_shape, bias, norm):
    """evaluate layerbatch centering"""
    # input_shape = uft.to_framework_channel(input_shape)
    xnp = np.arange(np.prod(input_shape)).reshape(input_shape)
    factory = None
    if norm:
        factory = SharedLipFactory()
    bn = BatchLipNorm(
        **{"num_features": size, "bias": bias, "normalize": norm, "factory": factory}
    )
    # bn_mom = bn.momentum
    if len(input_shape) == 2:
        mean_x = np.mean(xnp, axis=0)
        var_x = np.var(xnp, axis=0, ddof=1)
        mean_shape = (1, size)
    else:
        mean_x = np.mean(xnp, axis=(0, 2, 3))
        var_x = np.var(xnp, axis=(0, 2, 3), ddof=1)
        mean_shape = (1, size, 1, 1)
    x = uft.to_tensor(xnp)
    y = bn(x)
    scale_factor = current_scale_factor = 1.0
    if norm:
        scale_factor = 1.0 / np.max(np.sqrt(var_x))
    np.testing.assert_allclose(bn._get_mean(), mean_x, atol=1e-5)
    np.testing.assert_allclose(bn.get_scaling_factor(), scale_factor, atol=1e-5)
    np.testing.assert_allclose(
        uft.to_numpy(y),
        (xnp - np.reshape(mean_x, mean_shape)) * scale_factor,
        atol=1e-5,
    )
    y = bn(2 * x)
    new_runningmean = (
        mean_x + 2 * mean_x
    ) / 2.0  # mean_x * (1 - bn_mom) + 2 * mean_x * bn_mom
    # new_runningvar = (var_x + 4 * var_x)/2.0 #mean_x * (1 - bn_mom)
    # + 2 * mean_x * bn_mom
    conc_xnp = np.concatenate([xnp, 2 * xnp], axis=0)
    if len(input_shape) == 2:
        new_runningvar = np.var(conc_xnp, axis=0, ddof=1)
    else:
        new_runningvar = np.var(conc_xnp, axis=(0, 2, 3), ddof=1)
        # (var_x + 4 * var_x + 0.5*(mean_x-2 * mean_x)**2)/2.0 #mean_x * (1 - bn_mom)
        #  + 2 * mean_x * bn_mom
    if norm:
        current_scale_factor = 1.0 / np.max(np.sqrt(4 * var_x))
        scale_factor = 1.0 / np.max(np.sqrt(new_runningvar))
    np.testing.assert_allclose(bn._get_mean(), new_runningmean, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        bn.get_scaling_factor(), scale_factor, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        uft.to_numpy(y),
        (2 * xnp - 2 * np.reshape(mean_x, mean_shape)) * current_scale_factor,
        atol=1e-5,
        rtol=1e-5,
    )  # keep substract batch mean
    bn.eval()
    y = bn(2 * x)
    np.testing.assert_allclose(
        bn._get_mean(), new_runningmean, atol=1e-5
    )  # eval mode running mean freezed
    np.testing.assert_allclose(
        bn.get_scaling_factor(), scale_factor, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        uft.to_numpy(y),
        (2 * xnp - np.reshape(new_runningmean, mean_shape)) * scale_factor,
        atol=1e-5,
        rtol=1e-5,
    )  # eval mode use running_mean


@pytest.mark.parametrize(
    "norm_type",
    [
        BatchLipNorm,
    ],
)
@pytest.mark.parametrize(
    "norm",
    [False, True],
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (10, (10,), False),
        (10, (10,), True),
        (7, (7, 8, 8), False),
        (7, (7, 8, 8), True),
    ],
)
@pytest.mark.parametrize(
    "norm_factory",
    [False, True],
)
def test_batchlipnorm_serialization(
    norm_type, size, input_shape, bias, norm, norm_factory
):
    # Check serialization
    if hasattr(norm_type, "unavailable_class"):
        pytest.skip(f"{norm_type} not available")
    check_serialization(
        norm_type,
        layer_params={"num_features": size, "normalize": norm, "bias": bias},
        input_shape=input_shape,
        norm_factory=norm_factory and norm,
    )


@pytest.mark.parametrize(
    "norm_type",
    [
        BatchLipNorm,
    ],
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (10, (10,), True),
        (7, (7, 8, 8), True),
    ],
)
@pytest.mark.parametrize(
    "norm",
    [False, True],
)
@pytest.mark.parametrize(
    "norm_factory",
    [False, True],
)
def test_BatchLipNorm_bias(norm_type, size, input_shape, bias, norm, norm_factory):
    if hasattr(norm_type, "unavailable_class"):
        pytest.skip(f"{norm_type} not available")
    factory = None
    if norm and norm_factory:
        factory = SharedLipFactory()
    m = uft.generate_k_lip_model(
        norm_type,
        layer_params={
            "num_features": size,
            "bias": bias,
            "normalize": norm,
            "factory": factory,
        },
        input_shape=input_shape,
        k=1,
    )
    if m is None:
        pytest.skip()

    loss, optimizer, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=uft.CategoricalCrossentropy(from_logits=True),
    )
    batch_size = 10

    batch_size = 10
    bb = uft.to_numpy(uft.get_layer_by_index(m, 0).bias)
    sf = uft.to_numpy(uft.get_layer_by_index(m, 0).get_scaling_factor())
    np.testing.assert_allclose(bb, np.zeros((size,)), atol=1e-5)
    np.testing.assert_allclose(sf, 1.0, atol=1e-5)

    traind_ds = linear_generator(batch_size, input_shape)

    uft.train(
        traind_ds,
        m,
        loss,
        optimizer,
        2,
        batch_size,
        steps_per_epoch=10,
    )

    bb = uft.to_numpy(uft.get_layer_by_index(m, 0).bias)
    sf = uft.to_numpy(uft.get_layer_by_index(m, 0).get_scaling_factor())
    assert np.linalg.norm(bb) != 0.0
    if norm:
        assert np.linalg.norm(sf) != 1.0
    else:
        assert np.linalg.norm(sf) == 1.0


@pytest.mark.skipif(
    hasattr(BatchLipNorm, "unavailable_class"),
    reason="BatchLipNorm not available",
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (4, (3, 4), False),
        (4, (3, 4), True),
        (4, (3, 4, 8, 8), False),
        (4, (3, 4, 8, 8), True),
        (13, (64, 13, 8, 8), False),
        (13, (64, 13, 8, 8), True),
    ],
)
@pytest.mark.parametrize(
    "norm",
    [False, True],
)
@pytest.mark.parametrize(
    "type_seq",
    [0, 1, 2],
)
@pytest.mark.parametrize(
    "norm_factory",
    [False, True],
)
def test_BatchLipNorm_runningmean(
    size, input_shape, bias, norm, type_seq, norm_factory
):
    """evaluate batch centering convergence of running mean"""
    # input_shape = uft.to_framework_channel(input_shape)
    # start with 0 to set up running mean to zero
    if type_seq == 0:
        xnp = np.zeros(input_shape)
        gt_mean = 0.0
        gt_var = 1.0
        epochs = 2
    elif type_seq >= 1:
        epochs = 20
        xnp = np.random.normal(0.0, 1.0, input_shape)
        if len(input_shape) == 2:
            mean_x = np.mean(xnp, axis=0)
            var_x = np.var(xnp, axis=0, ddof=1)
            num_elem = input_shape[0]
        else:
            mean_x = np.mean(xnp, axis=(0, 2, 3))
            var_x = np.var(xnp, axis=(0, 2, 3), ddof=1)
            num_elem = input_shape[0] * input_shape[2] * input_shape[3]
        if type_seq == 1:
            gt_mean = mean_x
            gt_var = var_x * (num_elem - 1) * epochs / (epochs * num_elem - 1)
        else:
            conc_xnp = np.concatenate([xnp, 3 * xnp], axis=0)
            if len(input_shape) == 2:
                new_runningvar = np.var(conc_xnp, axis=0, ddof=1)
            else:
                new_runningvar = np.var(conc_xnp, axis=(0, 2, 3), ddof=1)
            gt_mean = (mean_x + 3 * mean_x) / 2.0
            print("gt mean_x ", mean_x, "new_runningvar ", new_runningvar)
            gt_var = (
                new_runningvar
                * (2 * num_elem - 1)
                * epochs
                / (2 * epochs * num_elem - 1)
            )

    factory = None
    if norm and norm_factory:
        factory = SharedLipFactory()

    bn = uft.get_instance_framework(
        BatchLipNorm,
        {"num_features": size, "bias": bias, "normalize": norm, "factory": factory},
    )

    x = uft.to_tensor(xnp)

    for _ in range(epochs):
        y = bn(x)  # noqa: F841
        if type_seq == 2:
            y = bn(3 * x)  # noqa: F841

    np.testing.assert_allclose(bn._get_mean(), gt_mean, atol=1e-5)
    # constant value => no scale factor
    if (type_seq == 0) or norm:
        print(
            "get_scaling_factor ",
            bn.get_scaling_factor(False),
            " var_x ",
            gt_var,
            1.0 / np.sqrt(np.max(gt_var)),
        )
        np.testing.assert_allclose(
            bn.get_scaling_factor(), 1.0 / np.sqrt(np.max(gt_var)), atol=1e-5
        )
