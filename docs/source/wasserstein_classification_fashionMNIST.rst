Example 4: HKR multiclass and fooling
=====================================

|Open in Colab|

This notebook will show how to train a Lispchitz network in a multiclass
configuration. The HKR (hinge-Kantorovich-Rubinstein) loss is extended
to multiclass using a one-vs all setup. The notebook will go through the
process of designing and training the network. It will also show how to
compute robustness certificates from the outputs of the network. Finally
the guarantee of these certificates will be checked by attacking the
network.

.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/deel-ai/deel-torchlip/blob/master/docs/notebooks/wasserstein_classification_fashionMNIST.ipynb

.. code:: ipython3

    # Install the required libraries deel-torchlip and foolbox (uncomment below if needed)
    # %pip install -qqq deel-torchlip foolbox

1. Data preparation
-------------------

For this example, the ``fashion_mnist`` dataset is used. In order to
keep things simple, no data augmentation is performed.

.. code:: ipython3

    import torch
    from torchvision import datasets, transforms
    
    train_set = datasets.FashionMNIST(
        root="./data",
        download=True,
        train=True,
        transform=transforms.ToTensor(),
    )
    
    test_set = datasets.FashionMNIST(
        root="./data",
        download=True,
        train=False,
        transform=transforms.ToTensor(),
    )
    
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size)


2. Model architecture
---------------------

The original one-vs-all setup would require 10 different networks (1 per
class). However, we use in practice a network with a common body and a
Lipschitz head (linear layer) containing 10 output neurons, like any
standard network for multiclass classification. Note that we use
torchlip.FrobeniusLinear disjoint_neurons=True to enforce each head
neuron to be a 1-Lipschitz function;

Notes about constraint enforcement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are currently 3 ways to enforce the Lipschitz constraint in a
network:

1. weight regularization
2. weight reparametrization
3. weight projection

Weight regularization doesn’t provide the required guarantees as it is
only a regularization term. Weight reparametrization is available in
``torchlip`` and is done directly in the layers (parameter
``niter_bjorck``). This trick allows to perform arbitrary gradient
updates without breaking the constraint. However this is done in the
graph, increasing resources consumption. Weight projection is not
implemented in ``torchlip``.

.. code:: ipython3

    from deel import torchlip
    
    # Sequential has the same properties as any Lipschitz layer. It only acts as a
    # container, with features specific to Lipschitz functions (condensation,
    # vanilla_exportation, ...)
    model = torchlip.Sequential(
        # Lipschitz layers preserve the API of their superclass (here Conv2d). An optional
        # argument is available, k_coef_lip, which controls the Lipschitz constant of the
        # layer
        torchlip.SpectralConv2d(
            in_channels=1, out_channels=16, kernel_size=(3, 3), padding="same"
        ),
        torchlip.GroupSort2(),
        # Usual pooling layer are implemented (avg, max), but new pooling layers are also
        # available
        torchlip.ScaledL2NormPool2d(kernel_size=(2, 2)),
        torchlip.SpectralConv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"
        ),
        torchlip.GroupSort2(),
        torchlip.ScaledL2NormPool2d(kernel_size=(2, 2)),
        # Our layers are fully interoperable with existing PyTorch layers
        torch.nn.Flatten(),
        torchlip.SpectralLinear(1568, 64),
        torchlip.GroupSort2(),
        torchlip.FrobeniusLinear(64, 10, bias=True, disjoint_neurons=True),
        # Similarly, model has a parameter to set the Lipschitz constant that automatically
        # sets the constant of each layer.
        k_coef_lip=1.0,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)





.. parsed-literal::

    Sequential(
      (0): ParametrizedSpectralConv2d(
        1, 16, kernel_size=(3, 3), stride=(1, 1), padding=same
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _SpectralNorm()
            (1): _BjorckNorm()
            (2): _LConvNorm()
          )
        )
      )
      (1): GroupSort2()
      (2): ScaledL2NormPool2d(norm_type=2, kernel_size=(2, 2), stride=None, ceil_mode=False)
      (3): ParametrizedSpectralConv2d(
        16, 32, kernel_size=(3, 3), stride=(1, 1), padding=same
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _SpectralNorm()
            (1): _BjorckNorm()
            (2): _LConvNorm()
          )
        )
      )
      (4): GroupSort2()
      (5): ScaledL2NormPool2d(norm_type=2, kernel_size=(2, 2), stride=None, ceil_mode=False)
      (6): Flatten(start_dim=1, end_dim=-1)
      (7): ParametrizedSpectralLinear(
        in_features=1568, out_features=64, bias=True
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _SpectralNorm()
            (1): _BjorckNorm()
          )
        )
      )
      (8): GroupSort2()
      (9): ParametrizedFrobeniusLinear(
        in_features=64, out_features=10, bias=True
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _FrobeniusNorm()
          )
        )
      )
    )



3. HKR loss and training
------------------------

The multiclass HKR loss can be found in the\ ``HKRMulticlassLoss``
class. The loss has two parameters: ``alpha`` and ``min_margin``.
Decreasing ``alpha`` and increasing ``min_margin`` improve robustness
(at the cost of accuracy). Note also in the case of Lipschitz networks,
more robustness requires more parameters. For more information, see `our
paper <https://arxiv.org/abs/2006.06520>`__.

In this setup, choosing ``alpha=0.99`` and ``min_margin=.25`` provides
good robustness without hurting the accuracy too much. An accurate
network can be obtained using ``alpha=0.999`` and ``min_margin=.1`` We
also propose the ``SoftHKRMulticlassLoss`` proposed in `this
paper <https://arxiv.org/abs/2206.06854>`__ that can achieve equivalent
performance to unconstrianed networks (92% validation accuracy with
``alpha=0.995``, ``min_margin=0.10``, ``temperature=50.0``). Finally the
``KRMulticlassLoss`` gives an indication on the robustness of the
network (proxy of the average certificate).

.. code:: ipython3

    loss_choice = "HKRMulticlassLoss" # "HKRMulticlassLoss" or "SoftHKRMulticlassLoss"
    epochs = 50
    
    optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())
    hkr_loss = None
    if loss_choice == "HKRMulticlassLoss":
        hkr_loss = torchlip.HKRMulticlassLoss(alpha=0.99, min_margin=0.25) #Robust
        #hkr_loss = torchlip.HKRMulticlassLoss(alpha=0.999, min_margin=0.10) #Accurate
    if loss_choice == "SoftHKRMulticlassLoss":
        hkr_loss = torchlip.SoftHKRMulticlassLoss(alpha=0.995, min_margin=0.10, temperature=50.0)
    assert hkr_loss is not None, "Please choose a valid loss function"
    
    kr_multiclass_loss = torchlip.KRMulticlassLoss()
    
    for epoch in range(epochs):
        m_kr, m_acc = 0, 0
    
        for step, (data, target) in enumerate(train_loader):
    
            # For multiclass HKR loss, the targets must be one-hot encoded
            target = torch.nn.functional.one_hot(target, num_classes=10)
            data, target = data.to(device), target.to(device)
    
            # Forward + backward pass
            optimizer.zero_grad()
            output = model(data)
            loss = hkr_loss(output, target)
            loss.backward()
            optimizer.step()
    
            # Compute metrics on batch
            m_kr += kr_multiclass_loss(output, target)
            m_acc += (output.argmax(dim=1) == target.argmax(dim=1)).sum() / len(target)
    
        # Train metrics for the current epoch
        metrics = [
            f"{k}: {v:.04f}"
            for k, v in {
                "loss": loss,
                "acc": m_acc / (step + 1),
                "KR": m_kr / (step + 1),
            }.items()
        ]
    
        # Compute validation loss for the current epoch
        test_output, test_targets = [], []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            test_output.append(model(data).detach().cpu())
            test_targets.append(
                torch.nn.functional.one_hot(target, num_classes=10).detach().cpu()
            )
        test_output = torch.cat(test_output)
        test_targets = torch.cat(test_targets)
    
        val_loss = hkr_loss(test_output, test_targets)
        val_kr = kr_multiclass_loss(test_output, test_targets)
        val_acc = (test_output.argmax(dim=1) == test_targets.argmax(dim=1)).float().mean()
    
        # Validation metrics for the current epoch
        metrics += [
            f"val_{k}: {v:.04f}"
            for k, v in {
                "loss": hkr_loss(test_output, test_targets),
                "acc": (test_output.argmax(dim=1) == test_targets.argmax(dim=1))
                .float()
                .mean(),
                "KR": kr_multiclass_loss(test_output, test_targets),
            }.items()
        ]
    
        print(f"Epoch {epoch + 1}/{epochs}")
        print(" - ".join(metrics))



.. parsed-literal::

    Epoch 1/50
    loss: 0.0161 - acc: 0.7948 - KR: 0.8425 - val_loss: 0.0234 - val_acc: 0.8237 - val_KR: 1.1728


.. parsed-literal::

    Epoch 2/50
    loss: 0.0144 - acc: 0.8425 - KR: 1.3253 - val_loss: 0.0181 - val_acc: 0.8474 - val_KR: 1.4679


.. parsed-literal::

    Epoch 3/50
    loss: 0.0040 - acc: 0.8522 - KR: 1.6386 - val_loss: 0.0191 - val_acc: 0.8214 - val_KR: 1.7816


.. parsed-literal::

    Epoch 4/50
    loss: 0.0046 - acc: 0.8574 - KR: 1.9427 - val_loss: 0.0098 - val_acc: 0.8596 - val_KR: 2.0056


.. parsed-literal::

    Epoch 5/50
    loss: 0.0000 - acc: 0.8605 - KR: 2.1595 - val_loss: 0.0079 - val_acc: 0.8680 - val_KR: 2.1441


.. parsed-literal::

    Epoch 6/50
    loss: 0.0049 - acc: 0.8642 - KR: 2.2765 - val_loss: 0.0063 - val_acc: 0.8634 - val_KR: 2.3429


.. parsed-literal::

    Epoch 7/50
    loss: -0.0053 - acc: 0.8670 - KR: 2.3516 - val_loss: 0.0051 - val_acc: 0.8664 - val_KR: 2.3691


.. parsed-literal::

    Epoch 8/50
    loss: -0.0021 - acc: 0.8708 - KR: 2.4078 - val_loss: 0.0031 - val_acc: 0.8698 - val_KR: 2.4568


.. parsed-literal::

    Epoch 9/50
    loss: -0.0072 - acc: 0.8731 - KR: 2.4747 - val_loss: 0.0031 - val_acc: 0.8688 - val_KR: 2.5106


.. parsed-literal::

    Epoch 10/50
    loss: 0.0009 - acc: 0.8726 - KR: 2.5210 - val_loss: 0.0026 - val_acc: 0.8685 - val_KR: 2.5051


.. parsed-literal::

    Epoch 11/50
    loss: -0.0028 - acc: 0.8751 - KR: 2.5462 - val_loss: 0.0022 - val_acc: 0.8730 - val_KR: 2.5741


.. parsed-literal::

    Epoch 12/50
    loss: -0.0035 - acc: 0.8751 - KR: 2.5864 - val_loss: 0.0025 - val_acc: 0.8707 - val_KR: 2.5648


.. parsed-literal::

    Epoch 13/50
    loss: -0.0027 - acc: 0.8764 - KR: 2.5977 - val_loss: 0.0019 - val_acc: 0.8718 - val_KR: 2.6368


.. parsed-literal::

    Epoch 14/50
    loss: -0.0047 - acc: 0.8789 - KR: 2.6347 - val_loss: 0.0044 - val_acc: 0.8539 - val_KR: 2.6234


.. parsed-literal::

    Epoch 15/50
    loss: 0.0189 - acc: 0.8788 - KR: 2.6543 - val_loss: 0.0003 - val_acc: 0.8723 - val_KR: 2.5902


.. parsed-literal::

    Epoch 16/50
    loss: 0.0142 - acc: 0.8793 - KR: 2.6534 - val_loss: 0.0006 - val_acc: 0.8673 - val_KR: 2.6843


.. parsed-literal::

    Epoch 17/50
    loss: -0.0018 - acc: 0.8809 - KR: 2.6729 - val_loss: 0.0014 - val_acc: 0.8670 - val_KR: 2.7061


.. parsed-literal::

    Epoch 18/50
    loss: 0.0005 - acc: 0.8805 - KR: 2.6892 - val_loss: 0.0002 - val_acc: 0.8692 - val_KR: 2.6683


.. parsed-literal::

    Epoch 19/50
    loss: 0.0144 - acc: 0.8814 - KR: 2.7032 - val_loss: 0.0006 - val_acc: 0.8754 - val_KR: 2.6909


.. parsed-literal::

    Epoch 20/50
    loss: 0.0095 - acc: 0.8827 - KR: 2.7164 - val_loss: 0.0001 - val_acc: 0.8707 - val_KR: 2.7713


.. parsed-literal::

    Epoch 21/50
    loss: -0.0062 - acc: 0.8815 - KR: 2.7312 - val_loss: -0.0008 - val_acc: 0.8776 - val_KR: 2.7397


.. parsed-literal::

    Epoch 22/50
    loss: -0.0057 - acc: 0.8834 - KR: 2.7449 - val_loss: -0.0002 - val_acc: 0.8638 - val_KR: 2.7346


.. parsed-literal::

    Epoch 23/50
    loss: -0.0109 - acc: 0.8844 - KR: 2.7543 - val_loss: -0.0016 - val_acc: 0.8781 - val_KR: 2.7080


.. parsed-literal::

    Epoch 24/50
    loss: -0.0091 - acc: 0.8844 - KR: 2.7597 - val_loss: -0.0006 - val_acc: 0.8731 - val_KR: 2.7509


.. parsed-literal::

    Epoch 25/50
    loss: 0.0054 - acc: 0.8839 - KR: 2.7827 - val_loss: -0.0021 - val_acc: 0.8789 - val_KR: 2.7414


.. parsed-literal::

    Epoch 26/50
    loss: -0.0093 - acc: 0.8865 - KR: 2.7827 - val_loss: -0.0024 - val_acc: 0.8815 - val_KR: 2.7571


.. parsed-literal::

    Epoch 27/50
    loss: -0.0028 - acc: 0.8854 - KR: 2.7891 - val_loss: -0.0007 - val_acc: 0.8671 - val_KR: 2.8054


.. parsed-literal::

    Epoch 28/50
    loss: 0.0045 - acc: 0.8848 - KR: 2.8087 - val_loss: -0.0005 - val_acc: 0.8765 - val_KR: 2.7992


.. parsed-literal::

    Epoch 29/50
    loss: -0.0050 - acc: 0.8855 - KR: 2.8126 - val_loss: -0.0003 - val_acc: 0.8716 - val_KR: 2.7960


.. parsed-literal::

    Epoch 30/50
    loss: -0.0090 - acc: 0.8858 - KR: 2.8186 - val_loss: -0.0015 - val_acc: 0.8727 - val_KR: 2.7698


.. parsed-literal::

    Epoch 31/50
    loss: -0.0086 - acc: 0.8882 - KR: 2.8209 - val_loss: -0.0029 - val_acc: 0.8752 - val_KR: 2.8335


.. parsed-literal::

    Epoch 32/50
    loss: -0.0064 - acc: 0.8871 - KR: 2.8258 - val_loss: -0.0030 - val_acc: 0.8820 - val_KR: 2.8266


.. parsed-literal::

    Epoch 33/50
    loss: -0.0086 - acc: 0.8882 - KR: 2.8410 - val_loss: -0.0025 - val_acc: 0.8742 - val_KR: 2.8252


.. parsed-literal::

    Epoch 34/50
    loss: -0.0157 - acc: 0.8873 - KR: 2.8518 - val_loss: -0.0021 - val_acc: 0.8736 - val_KR: 2.7995


.. parsed-literal::

    Epoch 35/50
    loss: 0.0009 - acc: 0.8877 - KR: 2.8418 - val_loss: -0.0028 - val_acc: 0.8739 - val_KR: 2.8467


.. parsed-literal::

    Epoch 36/50
    loss: -0.0137 - acc: 0.8882 - KR: 2.8552 - val_loss: -0.0023 - val_acc: 0.8778 - val_KR: 2.8063


.. parsed-literal::

    Epoch 37/50
    loss: -0.0103 - acc: 0.8881 - KR: 2.8597 - val_loss: -0.0023 - val_acc: 0.8720 - val_KR: 2.8331


.. parsed-literal::

    Epoch 38/50
    loss: -0.0100 - acc: 0.8897 - KR: 2.8594 - val_loss: -0.0033 - val_acc: 0.8811 - val_KR: 2.8638


.. parsed-literal::

    Epoch 39/50
    loss: -0.0047 - acc: 0.8887 - KR: 2.8630 - val_loss: -0.0035 - val_acc: 0.8801 - val_KR: 2.8755


.. parsed-literal::

    Epoch 40/50
    loss: -0.0047 - acc: 0.8902 - KR: 2.8691 - val_loss: -0.0023 - val_acc: 0.8752 - val_KR: 2.8752


.. parsed-literal::

    Epoch 41/50
    loss: -0.0085 - acc: 0.8897 - KR: 2.8753 - val_loss: -0.0018 - val_acc: 0.8756 - val_KR: 2.8190


.. parsed-literal::

    Epoch 42/50
    loss: -0.0170 - acc: 0.8892 - KR: 2.8745 - val_loss: -0.0034 - val_acc: 0.8807 - val_KR: 2.8524


.. parsed-literal::

    Epoch 43/50
    loss: -0.0025 - acc: 0.8909 - KR: 2.8805 - val_loss: -0.0030 - val_acc: 0.8811 - val_KR: 2.8388


.. parsed-literal::

    Epoch 44/50
    loss: -0.0093 - acc: 0.8922 - KR: 2.8824 - val_loss: -0.0034 - val_acc: 0.8805 - val_KR: 2.8573


.. parsed-literal::

    Epoch 45/50
    loss: -0.0065 - acc: 0.8898 - KR: 2.8861 - val_loss: -0.0027 - val_acc: 0.8763 - val_KR: 2.8508


.. parsed-literal::

    Epoch 46/50
    loss: -0.0046 - acc: 0.8908 - KR: 2.8799 - val_loss: -0.0038 - val_acc: 0.8808 - val_KR: 2.8540


.. parsed-literal::

    Epoch 47/50
    loss: -0.0141 - acc: 0.8902 - KR: 2.8932 - val_loss: -0.0037 - val_acc: 0.8794 - val_KR: 2.8714


.. parsed-literal::

    Epoch 48/50
    loss: -0.0101 - acc: 0.8912 - KR: 2.8959 - val_loss: -0.0033 - val_acc: 0.8789 - val_KR: 2.8827


.. parsed-literal::

    Epoch 49/50
    loss: -0.0111 - acc: 0.8918 - KR: 2.8873 - val_loss: -0.0040 - val_acc: 0.8859 - val_KR: 2.9193


.. parsed-literal::

    Epoch 50/50
    loss: -0.0008 - acc: 0.8933 - KR: 2.9104 - val_loss: -0.0041 - val_acc: 0.8818 - val_KR: 2.8705


4. Model export
---------------

Once training is finished, the model can be optimized for inference by
using the ``vanilla_export()`` method. The ``torchlip`` layers are
converted to their PyTorch counterparts, e.g. ``SpectralConv2d``
layers will be converted into ``torch.nn.Conv2d`` layers.

Warnings:
~~~~~~~~~

vanilla_export method modifies the model in-place.

In order to build and export a new model while keeping the reference
one, it is required to follow these steps:

# Build e new mode for instance with torchlip.Sequential(
torchlip.SpectralConv2d(…), …)

``vanilla_model = <your_function_to_build_the_model>()``

# Copy the parameters from the reference t the new model

``vanilla_model.load_state_dict(model.state_dict())``

# one forward required to initialize pamatrizations

``vanilla_model(one_input)``

# vanilla_export the new model

``vanilla_model = vanilla_model.vanilla_export()``

.. code:: ipython3

    vanilla_model = model.vanilla_export()
    vanilla_model.eval()
    vanilla_model.to(device)





.. parsed-literal::

    Sequential(
      (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (1): GroupSort2()
      (2): LPPool2d(norm_type=2, kernel_size=(2, 2), stride=None, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (4): GroupSort2()
      (5): LPPool2d(norm_type=2, kernel_size=(2, 2), stride=None, ceil_mode=False)
      (6): Flatten(start_dim=1, end_dim=-1)
      (7): Linear(in_features=1568, out_features=64, bias=True)
      (8): GroupSort2()
      (9): Linear(in_features=64, out_features=10, bias=True)
    )



5. Robustness evaluation: certificate generation and adversarial attacks
------------------------------------------------------------------------

A Lipschitz network provides certificates guaranteeing that there is no
adversarial attack smaller than the certificates. We will show how to
compute a certificate for a given image sample.

We will also run attacks on 10 images (one per class) and show that the
distance between the obtained adversarial images and the original images
is greater than the certificates. The ``foolbox`` library is used to
perform adversarial attacks.

.. code:: ipython3

    import numpy as np
    
    # Select only the first batch from the test set
    sub_data, sub_targets = next(iter(test_loader))
    sub_data, sub_targets = sub_data.to(device), sub_targets.to(device)
    
    # Drop misclassified elements
    output = vanilla_model(sub_data)
    well_classified_mask = output.argmax(dim=-1) == sub_targets
    sub_data = sub_data[well_classified_mask]
    sub_targets = sub_targets[well_classified_mask]
    
    # Retrieve one image per class
    images_list, targets_list = [], []
    for i in range(10):
        # Select the elements of the i-th label and keep the first one
        label_mask = sub_targets == i
        x = sub_data[label_mask][0]
        y = sub_targets[label_mask][0]
    
        images_list.append(x)
        targets_list.append(y)
    
    images = torch.stack(images_list)
    targets = torch.stack(targets_list)


In order to build a certificate :math:`\mathcal{M}` for a given sample,
we take the top-2 output and apply the following formula:

.. math::  \mathcal{M} = \frac{\text{top}_1 - \text{top}_2}{2} 

This certificate is a guarantee that no L2 attack can defeat the given
image sample with a robustness radius :math:`\epsilon` lower than the
certificate, i.e.

.. math::  \epsilon \geq \mathcal{M} 

In the following cell, we attack the model on the ten selected images
and compare the obtained radius :math:`\epsilon` with the certificates
:math:`\mathcal{M}`. In this setup, ``L2CarliniWagnerAttack`` from
``foolbox`` is used but in practice as these kind of networks are
gradient norm preserving, other attacks gives very similar results.

.. code:: ipython3

    import foolbox as fb
    
    # Compute certificates
    values, _ = vanilla_model(images).topk(k=2)
    #The factor is 2.0 when using disjoint_neurons==True
    certificates = (values[:, 0] - values[:, 1]) / 2. 
    
    # Run Carlini & Wagner attack
    fmodel = fb.PyTorchModel(vanilla_model, bounds=(0.0, 1.0), device=device)
    attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=6, steps=8000)
    _, advs, success = attack(fmodel, images, targets, epsilons=None)
    dist_to_adv = (images - advs).square().sum(dim=(1, 2, 3)).sqrt()
    
    # Print results
    print("Image #     Certificate     Distance to adversarial")
    print("---------------------------------------------------")
    for i in range(len(certificates)):
        print(f"Image {i}        {certificates[i]:.3f}                {dist_to_adv[i]:.2f}")



.. parsed-literal::

    Image #     Certificate     Distance to adversarial
    ---------------------------------------------------
    Image 0        0.309                1.29
    Image 1        1.864                4.65
    Image 2        0.397                1.56
    Image 3        0.527                2.81
    Image 4        0.105                0.44
    Image 5        0.188                0.82
    Image 6        0.053                0.26
    Image 7        0.450                1.62
    Image 8        1.488                3.91
    Image 9        0.161                0.69


Finally, we can take a visual look at the obtained images. When looking
at the adversarial examples, we can see that the network has interesting
properties:

-  **Predictability**: by looking at the certificates, we can predict if
   the adversarial example will be close or not to the original image.
-  **Disparity among classes**: as we can see, the attacks are very
   efficent on similar classes (e.g. T-shirt/top, and Shirt). This
   denotes that all classes are not made equal regarding robustness.
-  **Explainability**: the network is more explainable as attacks can be
   used as counterfactuals. We can tell that removing the inscription on
   a T-shirt turns it into a shirt makes sense. Non-robust examples
   reveal that the network relies on textures rather on shapes to make
   its decision.

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    def adversarial_viz(model, images, advs, class_names):
        """
        This functions shows for each image sample:
        - the original image
        - the adversarial image
        - the difference map
        - the certificate and the observed distance to adversarial
        """
        scale = 1.5
        nb_imgs = images.shape[0]
    
        # Compute certificates
        values, _ = model(images).topk(k=2)
        certificates = (values[:, 0] - values[:, 1]) / np.sqrt(2)
    
        # Compute distance between image and its adversarial
        dist_to_adv = (images - advs).square().sum(dim=(1, 2, 3)).sqrt()
    
        # Find predicted classes for images and their adversarials
        orig_classes = [class_names[i] for i in model(images).argmax(dim=-1)]
        advs_classes = [class_names[i] for i in model(advs).argmax(dim=-1)]
    
        # Compute difference maps
        advs = advs.detach().cpu()
        images = images.detach().cpu()
        diff_pos = np.clip(advs - images, 0, 1.0)
        diff_neg = np.clip(images - advs, 0, 1.0)
        diff_map = np.concatenate(
            [diff_neg, diff_pos, np.zeros_like(diff_neg)], axis=1
        ).transpose((0, 2, 3, 1))
    
        # Create plot
        def _set_ax(ax, title):
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
    
        figsize = (3 * scale, nb_imgs * scale)
        _, axes = plt.subplots(
            ncols=3, nrows=nb_imgs, figsize=figsize, squeeze=False, constrained_layout=True
        )
        for i in range(nb_imgs):
            _set_ax(axes[i][0], orig_classes[i])
            axes[i][0].imshow(images[i].squeeze(), cmap="gray")
            _set_ax(axes[i][1], advs_classes[i])
            axes[i][1].imshow(advs[i].squeeze(), cmap="gray")
            _set_ax(axes[i][2], f"certif: {certificates[i]:.2f}, obs: {dist_to_adv[i]:.2f}")
            axes[i][2].imshow(diff_map[i] / diff_map[i].max())
    
    
    adversarial_viz(vanilla_model, images, advs, test_set.classes)




.. image:: wasserstein_classification_fashionMNIST_files/wasserstein_classification_fashionMNIST_16_0.png

