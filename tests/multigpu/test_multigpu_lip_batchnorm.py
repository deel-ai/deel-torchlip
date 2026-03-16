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

import numpy as np
import torch
import torch.utils


from deel.torchlip.modules.normalization import BatchLipNorm, SharedLipFactory

import torch.distributed as dist

device = "cuda"


class ModelBN(torch.nn.Module):
    # Our model

    def __init__(self, num_features, bias, normalize, factory):
        super(ModelBN, self).__init__()
        self.bn = BatchLipNorm(
            **{
                "num_features": num_features,
                "bias": bias,
                "normalize": normalize,
                "factory": factory,
            }
        )
        self.avg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(num_features, num_features)

    def forward(self, input):
        output = self.bn(input)
        # print("\tIn Model: input size", input.size(),
        #      "output size", output.size())
        if len(output.size()) > 2:
            output2 = self.avg(output)
            output2 = self.linear(output2.view(output2.size(0), -1))
            output2 = output2.view(output2.size(0), -1, 1, 1)
        else:
            output2 = self.linear(output)

        return (
            output2 + output - output2
        )  # just to have a different output shape for the test


class RandomDataset(torch.utils.data.Dataset):

    def __init__(self, input_shape, type_seq=-1):
        self.input_shape = input_shape
        np.random.seed(42)
        if type_seq == 0:
            self.data = np.zeros(input_shape, dtype=np.float32)
        elif type_seq == 1:
            self.data = np.random.normal(0.0, 1.0, input_shape).astype(np.float32)
        elif type_seq == 2:
            self.data = np.random.normal(0.0, 1.0, input_shape).astype(np.float32)
            self.data = np.concatenate([self.data, 3 * self.data], axis=0)
            self.input_shape = (2 * input_shape[0],) + input_shape[1:]
        else:
            self.data = np.arange(np.prod(input_shape)).reshape(input_shape)

    def __getitem__(self, index):
        return self.data[index].astype(np.float32)

    def __len__(self):
        return self.input_shape[0]


def test_BatchLipNorm(size, input_shape, bias, norm, device="cpu"):
    """evaluate layerbatch centering"""
    dataset = RandomDataset(input_shape)
    xnp = dataset.data
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    rand_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=input_shape[0], shuffle=False, sampler=sampler
    )

    factory = None
    if norm:
        factory = SharedLipFactory()

    bn = ModelBN(
        **{"num_features": size, "bias": bias, "normalize": norm, "factory": factory}
    ).to(device)
    mbn_src = bn
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        bn = torch.nn.parallel.DistributedDataParallel(bn)
        mbn_src = bn.module

    bn = bn.to(device)
    # bn_mom = bn.momentum
    if len(input_shape) == 2:
        mean_x = np.mean(xnp, axis=0)
        var_x = np.var(xnp, axis=0, ddof=1)
        mean_shape = (1, size)
    else:
        mean_x = np.mean(xnp, axis=(0, 2, 3))
        var_x = np.var(xnp, axis=(0, 2, 3), ddof=1)
        mean_shape = (1, size, 1, 1)

    x = next(iter(rand_loader)).to(device)
    y = bn(x)
    scale_factor = 1.0  # current_scale_factor = 1.0
    if norm:
        scale_factor = 1.0 / np.max(np.sqrt(var_x))

    np.testing.assert_allclose(
        mbn_src.bn._get_mean(training=False).detach().cpu().numpy(),
        mean_x,
        atol=1e-4,
        rtol=1e-4,
    )

    np.testing.assert_allclose(
        mbn_src.bn.get_scaling_factor().detach().cpu().numpy(),
        scale_factor,
        atol=1e-4,
        rtol=1e-4,
    )

    y = bn(2 * x)
    new_runningmean = (
        mean_x + 2 * mean_x
    ) / 2.0  # mean_x * (1 - bn_mom) + 2 * mean_x * bn_mom

    conc_xnp = np.concatenate([xnp, 2 * xnp], axis=0)
    if len(input_shape) == 2:
        new_runningvar = np.var(conc_xnp, axis=0, ddof=1)
    else:
        new_runningvar = np.var(conc_xnp, axis=(0, 2, 3), ddof=1)

    if norm:
        # current_scale_factor = 1.0 / np.max(np.sqrt(4 * var_x))
        scale_factor = 1.0 / np.max(np.sqrt(new_runningvar))
    np.testing.assert_allclose(
        mbn_src.bn._get_mean(training=False).detach().cpu().numpy(),
        new_runningmean,
        atol=1e-4,
        rtol=1e-4,
    )

    np.testing.assert_allclose(
        mbn_src.bn.get_scaling_factor().detach().cpu().numpy(),
        scale_factor,
        atol=1e-4,
        rtol=1e-4,
    )

    bn.eval()
    y = bn(2 * x)
    np.testing.assert_allclose(
        mbn_src.bn._get_mean(training=False).detach().cpu().numpy(),
        new_runningmean,
        atol=1e-4,
        rtol=1e-4,
    )  # eval mode running mean freezed
    np.testing.assert_allclose(
        mbn_src.bn.get_scaling_factor().detach().cpu().numpy(),
        scale_factor,
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        y.detach().cpu().numpy(),
        (2 * x.detach().cpu().numpy() - np.reshape(new_runningmean, mean_shape))
        * scale_factor,
        atol=1e-4,
        rtol=1e-4,
    )  # eval mode use running_mean
    print("OK")


def test_BatchLipNorm3(size, input_shape, bias, norm, device="cpu"):
    """evaluate layerbatch centering"""

    dataset = RandomDataset(input_shape)
    xnp = dataset.data
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    rand_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=input_shape[0], shuffle=False, sampler=sampler
    )

    factory = None
    if norm:
        factory = SharedLipFactory()

    bn = ModelBN(
        **{"num_features": size, "bias": bias, "normalize": norm, "factory": factory}
    ).to(device)
    mbn_src = bn
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        bn = torch.nn.parallel.DistributedDataParallel(bn)
        mbn_src = bn.module

    bn = bn.to(device)
    # bn_mom = bn.momentum
    if len(input_shape) == 2:
        mean_x = np.mean(xnp, axis=0)
        # var_x = np.var(xnp, axis=0, ddof=1)
        mean_shape = (1, size)
    else:
        mean_x = np.mean(xnp, axis=(0, 2, 3))
        # var_x = np.var(xnp, axis=(0, 2, 3), ddof=1)
        mean_shape = (1, size, 1, 1)

    scale_factor = 1.0  # current_scale_factor = 1.0
    x = next(iter(rand_loader)).to(device)
    y = bn(x)
    y = bn(0.5 * x)
    y = bn(0.25 * x)

    new_runningmean = (
        mean_x + 0.5 * mean_x + 0.25 * mean_x
    ) / 3.0  # mean_x * (1 - bn_mom) + 2 * mean_x * bn_mom

    conc_xnp = np.concatenate([xnp, 0.5 * xnp, 0.25 * xnp], axis=0)
    if len(input_shape) == 2:
        new_runningvar = np.var(conc_xnp, axis=0, ddof=1)
    else:
        new_runningvar = np.var(conc_xnp, axis=(0, 2, 3), ddof=1)

    if norm:
        # current_scale_factor = 1.0 / np.max(np.sqrt(4 * var_x))
        scale_factor = 1.0 / np.max(np.sqrt(new_runningvar))
    np.testing.assert_allclose(
        mbn_src.bn._get_mean(training=False).detach().cpu().numpy(),
        new_runningmean,
        atol=1e-4,
        rtol=1e-4,
    )

    np.testing.assert_allclose(
        mbn_src.bn.get_scaling_factor().detach().cpu().numpy(),
        scale_factor,
        atol=1e-4,
        rtol=1e-4,
    )

    bn.eval()
    y = bn(2 * x)
    np.testing.assert_allclose(
        mbn_src.bn._get_mean(training=False).detach().cpu().numpy(),
        new_runningmean,
        atol=1e-4,
        rtol=1e-4,
    )  # eval mode running mean freezed
    np.testing.assert_allclose(
        mbn_src.bn.get_scaling_factor().detach().cpu().numpy(),
        scale_factor,
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        y.detach().cpu().numpy(),
        (2 * x.detach().cpu().numpy() - np.reshape(new_runningmean, mean_shape))
        * scale_factor,
        atol=1e-3,
        rtol=1e-3,
    )  # eval mode use running_mean
    print("OK")


def test_BatchLipNorm_runningmean(size, input_shape, bias, norm, type_seq):
    """evaluate batch centering convergence of running mean"""
    # input_shape = uft.to_framework_channel(input_shape)
    # start with 0 to set up running mean to zero
    dataset = RandomDataset(input_shape, type_seq=type_seq)
    xnp = dataset.data
    sampler = torch.utils.data.DistributedSampler(dataset)
    rand_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=input_shape[0], shuffle=False, sampler=sampler
    )
    if type_seq == 0:
        gt_mean = 0.0
        gt_var = 1.0
        epochs = 2
    elif type_seq >= 1:
        epochs = 20
        if len(input_shape) == 2:
            mean_x = np.mean(xnp, axis=0)
            var_x = np.var(xnp, axis=0, ddof=1)
            num_elem = xnp.shape[0]
        else:
            mean_x = np.mean(xnp, axis=(0, 2, 3))
            var_x = np.var(xnp, axis=(0, 2, 3), ddof=1)
            num_elem = xnp.shape[0] * xnp.shape[2] * xnp.shape[3]

        gt_mean = mean_x  # concat is done in RandomDataset
        gt_var = var_x * (num_elem - 1) * epochs / (epochs * num_elem - 1)

    factory = None
    if norm:
        factory = SharedLipFactory()

    bn = ModelBN(
        **{"num_features": size, "bias": bias, "normalize": norm, "factory": factory}
    ).to(device)
    mbn_src = bn
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        bn = torch.nn.parallel.DistributedDataParallel(bn)
        mbn_src = bn.module

    bn = bn.to(device)

    print(f"running {epochs} epochs")
    for _ in range(epochs):
        for x in rand_loader:
            # print("x shape", x.shape)
            x = x.to(device)
            y = bn(x)  # noqa: F841
    # noqa: F841

    np.testing.assert_allclose(
        mbn_src.bn._get_mean(training=False).detach().cpu().numpy(), gt_mean, atol=1e-5
    )
    # constant value => no scale factor
    if (type_seq == 0) or norm:
        np.testing.assert_allclose(
            mbn_src.bn.get_scaling_factor().detach().cpu().numpy(),
            1.0 / np.sqrt(np.max(gt_var)),
            atol=1e-5,
        )


if __name__ == "__main__":
    print(
        "Usage: torchrun --nproc_per_node=NPROC  "
        "tests/multigpu/test_multigpu_lip_batchnorm.py"
    )
    print("start test_BatchLipNorm")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")
    torch.cuda.set_device(local_rank)

    configs_test = [
        (4, (128, 4), False, False),
        (4, (128, 4), False, True),
        (4, (128, 4), True, False),
        (4, (128, 4), True, True),
        (4, (128, 4, 8, 8), False, False),
        (4, (128, 4, 8, 8), False, True),
        (4, (128, 4, 8, 8), True, False),
        (4, (128, 4, 8, 8), True, True),
    ]
    for ii, (size, input_shape, bias, norm) in enumerate(configs_test):
        print(
            f"start test_BatchLipNorm3 {ii+1},: size={size}, "
            f"input_shape={input_shape}, bias={bias}, norm={norm}"
        )
        test_BatchLipNorm3(size, input_shape, bias, norm, device)
        print(f"test {ii+1} OK")

    configs_test = [
        (4, (128, 4), False, False),
        (4, (128, 4), False, True),
        (4, (128, 4), True, False),
        (4, (128, 4), True, True),
        (4, (128, 4, 8, 8), False, False),
        (4, (128, 4, 8, 8), False, True),
        (4, (128, 4, 8, 8), True, False),
        (4, (128, 4, 8, 8), True, True),
    ]
    for ii, (size, input_shape, bias, norm) in enumerate(configs_test):
        print(
            f"start test_BatchLipNorm {ii+1},: size={size}, "
            f"input_shape={input_shape}, bias={bias}, norm={norm}"
        )
        test_BatchLipNorm(size, input_shape, bias, norm, device)
        print(f"test {ii+1} OK")

    config2_test = [
        (4, (128, 4), False),
        (4, (128, 4), True),
        (4, (128, 4, 8, 8), False),
        (4, (128, 4, 8, 8), True),
        (13, (64, 13, 8, 8), False),
        (13, (64, 13, 8, 8), True),
    ]
    config2_norm_test = [False, True]
    config2_type_test = [0, 1, 2]
    num_test = 0
    for type_seq in config2_type_test:
        for norm in config2_norm_test:
            for size, input_shape, bias in config2_test:
                num_test += 1
                print(
                    f"start test_BatchLipNorm_runningmean {num_test},: "
                    f"size={size}, input_shape={input_shape}, bias={bias}, "
                    f"norm={norm}, type_seq={type_seq}"
                )
                test_BatchLipNorm_runningmean(size, input_shape, bias, norm, type_seq)
                print(f"test {num_test} OK")
