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

from typing import Optional, Union
import torch

if torch.__version__.startswith("1."):
    import functorch as tfc
else:
    import torch.func as tfc


def evaluate_lip_const(
    model: torch.nn.Module,
    x: Optional[torch.Tensor] = None,  # can be torch tensor or None
    evaluation_type: Union[str, list[str]] = "jacobian_norm",
    disjoint_neurons=False,
    input_shape=None,
    expected_value=None,
    **kwargs,
) -> float:
    """
    Evaluate the Lipschitz constant of a model, using different methods.
    Please note that the estimation of the lipschitz constant is done locally around
    input samples. This may not correctly estimate the behaviour in the whole domain.
    Args:
        model: built torch model used to make predictions
        x: inputs used to compute the lipschitz constant. If None, input_shape must be
            provided to generate random inputs. (shape: (batch_size, ...))
        evaluation_type: method used to evaluate the lipschitz constant. Can be one of
            "jacobian_norm", "noise_norm", "attack" or "all".
        disjoint_neurons: if True, each output neuron is considered as a separate
            1-Lipschitz function.
        input_shape: shape of the input tensor, used if x is None.
        expected_value: if provided, the computed lipschitz constant is compared to
            this value and an assertion error is raised if the computed value is higher
            (warning the assertion is strict).
    """
    type2fct = {
        "jacobian_norm": evaluate_lip_const_jacobian_norm,
        "noise_norm": evaluate_lip_const_noise_norm,
        "attack": evaluate_lip_const_attack,
    }
    device = "cpu"
    if len(list(model.parameters())) > 0:
        device = next(model.parameters()).device

    if evaluation_type == "all":
        evaluation_type = list(type2fct.keys())
    if isinstance(evaluation_type, list):
        lip_csts = []
        for etype in evaluation_type:
            lip_csts.append(
                evaluate_lip_const(
                    model,
                    x,
                    etype,
                    disjoint_neurons,
                    input_shape,
                    expected_value,
                    **kwargs,
                )
            )
        return float(torch.max(torch.tensor(lip_csts)).item())

    if evaluation_type not in type2fct:
        raise ValueError(
            f"Unknown evaluation_type {evaluation_type}. "
            f"Available types are {list(type2fct.keys())}"
        )
    if x is None:
        if input_shape is None:
            raise ValueError("If x is None, input_shape must be provided")
        x = torch.randn(input_shape)
    x = x.to(device)
    val_lip_const = type2fct[evaluation_type](model, x, disjoint_neurons, **kwargs)
    print(
        f"Empirical lipschitz constant is {val_lip_const} with method {evaluation_type}"
    )
    if expected_value is not None:
        assert (
            val_lip_const <= expected_value
        ), f"Empirical lipschitz constant {val_lip_const} is \
            higher than expected value {expected_value}"
    return val_lip_const


def evaluate_lip_const_noise_norm(
    model: torch.nn.Module,
    x: torch.Tensor,
    disjoint_neurons=False,
    num_noisy_samples: int = 10,
    epsilon_noise: float = 1.0,
    **kwargs,
) -> float:
    """
    Evaluate the Lipschitz constant of a model, using random noise added to the input.
    Please note that the estimation of the lipschitz constant is done locally around
    input samples. This may not correctly estimate the behaviour in the whole domain.
    Args:
        model: built torch model used to make predictions
        x: inputs used to compute the lipschitz constant (shape: (batch_size, ...))
        disjoint_neurons: if True, each output neuron is considered as a separate
            1-Lipschitz function.
        num_noisy_samples: number of random noise samples to use for the estimation
        epsilon_noise: standard deviation of the gaussian noise added to the input
    Returns:
        float: the empirically evaluated Lipschitz constant (max over batch).
    """
    with torch.no_grad():
        model.eval()
        pred = model(x)
        lip_csts = []
        for _ in range(num_noisy_samples):  # random sampling
            noise = epsilon_noise * torch.randn_like(x) * torch.rand(1).to(x.device)
            noisy_input = x + noise
            noisy_pred = model(noisy_input)
            if not disjoint_neurons:
                pred_diff_norm = torch.linalg.norm(
                    (pred - noisy_pred).view(pred.shape[0], -1), dim=1
                )
            else:
                # each output neuron is a 1Lipschitz function
                diff_pred = pred - noisy_pred
                diff_pred = diff_pred.view(diff_pred.shape[0], -1, diff_pred.shape[-1])
                pred_diff_norm = torch.linalg.norm(diff_pred, dim=1)
                pred_diff_norm = torch.max(pred_diff_norm, dim=-1).values

            input_diff_norm = torch.linalg.norm(noise.view(pred.shape[0], -1), dim=1)
            lip_cst = pred_diff_norm / input_diff_norm
            lip_csts.append(lip_cst)
        lip_csts = torch.cat(lip_csts, dim=0)
        return float(torch.max(lip_csts).item())


def evaluate_lip_const_jacobian_norm(
    model: torch.nn.Module,
    x: torch.Tensor,
    disjoint_neurons=False,
    **kwargs,
) -> float:
    """
    Evaluate the Lipschitz constant of a model, using the Jacobian of the model.
    Please note that the estimation of the lipschitz constant is done locally around
    input samples. This may not correctly estimate the behaviour in the whole domain.

    Args:
        model: built torch model used to make predictions
        x: inputs used to compute the lipschitz constant
        disjoint_neurons: if True, each output neuron is considered as a separate
            1-Lipschitz function.

    Returns:
        float: the empirically evaluated Lipschitz constant. The computation might also
            be inaccurate in high dimensional space.

    """

    # Define a function that computes the model output
    def model_func(x):
        # using vmap torchfunc method induce a single sample input
        # so we need to unsqueeze the input
        y = model(torch.unsqueeze(x, dim=0))  # Forward pass
        return y

    # assert disjoint_neurons is False, "disjoint_neurons=True not implemented yet"
    x_src = x.clone().detach().requires_grad_(True)

    # Compute the Jacobian using jacrev
    batch_jacobian = tfc.vmap(tfc.jacrev(model_func))(x_src)
    # Reshape the Jacobian to match the desired shape
    batch_size = x.shape[0]
    xdim = torch.prod(torch.tensor(x.shape[1:])).item()

    if not disjoint_neurons:
        batch_jacobian = batch_jacobian.view(batch_size, -1, xdim)
    else:
        # each output neuron is a 1Lipschitz function:
        # compute the norm of each output neuron
        outdim = len(batch_jacobian.shape) - len(x.shape[1:]) - 1
        outsize = batch_jacobian.shape[outdim]
        # switch outdim to be the first dimension
        batch_jacobian = batch_jacobian.moveaxis(outdim, 0)
        batch_jacobian = batch_jacobian.reshape(outsize * batch_size, -1, xdim)

    # Compute singular values and check Lipschitz property
    lip_cst = torch.linalg.norm(batch_jacobian, ord=2, dim=(-2, -1))
    return float(torch.max(lip_cst).item())


def _compute_disjoint_neurons_lip_const(
    ref_output: torch.Tensor, noisy_pred: torch.Tensor
) -> torch.Tensor:
    # each output neuron is a 1Lipschitz function: attack the maximum
    diff_pred = ref_output - noisy_pred
    diff_pred = diff_pred.view(diff_pred.shape[0], diff_pred.shape[1], -1)
    pred_diff_norm = torch.linalg.norm(diff_pred, dim=-1)
    pred_diff_norm = torch.max(pred_diff_norm, dim=1).values
    return pred_diff_norm


def evaluate_lip_const_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    disjoint_neurons=False,
    num_iterations: int = 100,
    step_size: float = 1e-2,
    double_attack: bool = False,
    **kwargs,
) -> float:
    """
    Evaluate the Lipschitz constant of a model, using an adversarial attack.
    Please note that the estimation of the lipschitz constant is done locally around
    input samples. This may not correctly estimate the behaviour in the whole domain.

    Args:
        model: built torch model used to make predictions
        x: inputs used to compute the lipschitz constant
        disjoint_neurons: if True, each output neuron is considered as a separate
            1-Lipschitz function.
        num_iterations: number of iterations for the attack
        step_size: step size for each iteration
        double_attack: if True, perform a second attack starting from the worst case
            found during the first attack. This can improve the estimation of the
            Lipschitz constant at the cost of doubling the computation time.

    Returns:
        float: the empirically evaluated Lipschitz constant. The computation might also
            be inaccurate in high dimensional space.

    """

    def attack_lip_const(model, x, step_size, num_iterations):
        model.eval()
        with torch.no_grad():
            ref_output = model(x).detach()
        noise = (torch.randn_like(x) * torch.rand(1).to(x.device)).detach()
        for _ in range(num_iterations):
            noise = noise.requires_grad_(True)
            noisy_input = x + noise
            noisy_pred = model(noisy_input)

            if not disjoint_neurons:
                pred_diff_norm = torch.linalg.norm(
                    (ref_output - noisy_pred).view(ref_output.shape[0], -1), dim=1
                )
            else:
                # each output neuron is a 1Lipschitz function: attack the maximum
                pred_diff_norm = _compute_disjoint_neurons_lip_const(
                    ref_output, noisy_pred
                )
            input_diff_norm = torch.linalg.norm(
                noise.view(ref_output.shape[0], -1), dim=1
            )
            lip_cst = pred_diff_norm / input_diff_norm
            loss = lip_cst.mean()

            grad = torch.autograd.grad(loss, noise)[0]
            noise = noise + step_size * grad.sign()
            noise = noise.detach()

        noisy_input = x + noise
        noisy_pred = model(noisy_input)
        if not disjoint_neurons:
            pred_diff_norm = torch.linalg.norm(
                (ref_output - noisy_pred).view(ref_output.shape[0], -1), dim=1
            )
        else:
            # each output neuron is a 1Lipschitz function: attack the maximum
            pred_diff_norm = _compute_disjoint_neurons_lip_const(ref_output, noisy_pred)
        input_diff_norm = torch.linalg.norm(noise.view(ref_output.shape[0], -1), dim=1)
        lip_cst = pred_diff_norm / input_diff_norm
        return noise, lip_cst

    if double_attack:
        print("Warning : double_attack is set to True, \
                the computation time will be doubled")
    noise, lip_cst = attack_lip_const(model, x, step_size, num_iterations)

    if not double_attack:
        return float(torch.max(lip_cst).item())

    # double attack

    noise2, lip_cst2 = attack_lip_const(model, x + noise, step_size, num_iterations)
    return float(torch.max(torch.cat([lip_cst, lip_cst2], dim=0)).item())
