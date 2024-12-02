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
      (2): ScaledL2NormPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
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
      (5): ScaledL2NormPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
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
    loss: 0.0193 - acc: 0.7896 - KR: 0.8442 - val_loss: 0.0213 - val_acc: 0.8244 - val_KR: 1.2169


.. parsed-literal::

    Epoch 2/50
    loss: 0.0124 - acc: 0.8482 - KR: 1.4474 - val_loss: 0.0186 - val_acc: 0.8342 - val_KR: 1.6805


.. parsed-literal::

    Epoch 3/50
    loss: 0.0109 - acc: 0.8542 - KR: 1.8511 - val_loss: 0.0118 - val_acc: 0.8538 - val_KR: 2.0030


.. parsed-literal::

    Epoch 4/50
    loss: 0.0060 - acc: 0.8587 - KR: 2.1384 - val_loss: 0.0072 - val_acc: 0.8534 - val_KR: 2.2039


.. parsed-literal::

    Epoch 5/50
    loss: 0.0019 - acc: 0.8619 - KR: 2.2898 - val_loss: 0.0088 - val_acc: 0.8419 - val_KR: 2.3712


.. parsed-literal::

    Epoch 6/50
    loss: 0.0062 - acc: 0.8658 - KR: 2.3825 - val_loss: 0.0049 - val_acc: 0.8675 - val_KR: 2.4397


.. parsed-literal::

    Epoch 7/50
    loss: 0.0162 - acc: 0.8681 - KR: 2.4547 - val_loss: 0.0041 - val_acc: 0.8647 - val_KR: 2.4717


.. parsed-literal::

    Epoch 8/50
    loss: 0.0046 - acc: 0.8709 - KR: 2.4912 - val_loss: 0.0042 - val_acc: 0.8645 - val_KR: 2.4645


.. parsed-literal::

    Epoch 9/50
    loss: -0.0095 - acc: 0.8717 - KR: 2.5289 - val_loss: 0.0027 - val_acc: 0.8713 - val_KR: 2.5118


.. parsed-literal::

    Epoch 10/50
    loss: 0.0066 - acc: 0.8751 - KR: 2.5463 - val_loss: 0.0048 - val_acc: 0.8578 - val_KR: 2.6126


.. parsed-literal::

    Epoch 11/50
    loss: 0.0102 - acc: 0.8746 - KR: 2.5673 - val_loss: 0.0039 - val_acc: 0.8673 - val_KR: 2.5540


.. parsed-literal::

    Epoch 12/50
    loss: -0.0033 - acc: 0.8756 - KR: 2.5913 - val_loss: 0.0020 - val_acc: 0.8648 - val_KR: 2.5890


.. parsed-literal::

    Epoch 13/50
    loss: -0.0091 - acc: 0.8775 - KR: 2.6237 - val_loss: 0.0025 - val_acc: 0.8708 - val_KR: 2.5836


.. parsed-literal::

    Epoch 14/50
    loss: -0.0021 - acc: 0.8780 - KR: 2.6263 - val_loss: 0.0030 - val_acc: 0.8583 - val_KR: 2.6685


.. parsed-literal::

    Epoch 15/50
    loss: 0.0211 - acc: 0.8785 - KR: 2.6446 - val_loss: 0.0027 - val_acc: 0.8595 - val_KR: 2.6300


.. parsed-literal::

    Epoch 16/50
    loss: 0.0062 - acc: 0.8789 - KR: 2.6743 - val_loss: 0.0016 - val_acc: 0.8634 - val_KR: 2.6763


.. parsed-literal::

    Epoch 17/50
    loss: -0.0101 - acc: 0.8805 - KR: 2.7005 - val_loss: -0.0009 - val_acc: 0.8766 - val_KR: 2.6881


.. parsed-literal::

    Epoch 18/50
    loss: 0.0014 - acc: 0.8831 - KR: 2.7211 - val_loss: -0.0007 - val_acc: 0.8783 - val_KR: 2.7363


.. parsed-literal::

    Epoch 19/50
    loss: -0.0027 - acc: 0.8812 - KR: 2.7439 - val_loss: -0.0001 - val_acc: 0.8708 - val_KR: 2.7713


.. parsed-literal::

    Epoch 20/50
    loss: -0.0044 - acc: 0.8835 - KR: 2.7603 - val_loss: -0.0002 - val_acc: 0.8716 - val_KR: 2.7494


.. parsed-literal::

    Epoch 21/50
    loss: -0.0117 - acc: 0.8837 - KR: 2.7681 - val_loss: 0.0012 - val_acc: 0.8702 - val_KR: 2.7200


.. parsed-literal::

    Epoch 22/50
    loss: -0.0140 - acc: 0.8844 - KR: 2.7766 - val_loss: -0.0014 - val_acc: 0.8782 - val_KR: 2.8377


.. parsed-literal::

    Epoch 23/50
    loss: -0.0074 - acc: 0.8863 - KR: 2.7910 - val_loss: 0.0004 - val_acc: 0.8747 - val_KR: 2.7969


.. parsed-literal::

    Epoch 24/50
    loss: -0.0056 - acc: 0.8868 - KR: 2.7963 - val_loss: -0.0002 - val_acc: 0.8682 - val_KR: 2.7982


.. parsed-literal::

    Epoch 25/50
    loss: -0.0092 - acc: 0.8870 - KR: 2.7979 - val_loss: -0.0025 - val_acc: 0.8808 - val_KR: 2.8081


.. parsed-literal::

    Epoch 26/50
    loss: 0.0144 - acc: 0.8869 - KR: 2.8073 - val_loss: -0.0016 - val_acc: 0.8783 - val_KR: 2.8037


.. parsed-literal::

    Epoch 27/50
    loss: -0.0063 - acc: 0.8887 - KR: 2.8083 - val_loss: -0.0020 - val_acc: 0.8793 - val_KR: 2.7780


.. parsed-literal::

    Epoch 28/50
    loss: -0.0097 - acc: 0.8886 - KR: 2.8210 - val_loss: -0.0003 - val_acc: 0.8742 - val_KR: 2.7555


.. parsed-literal::

    Epoch 29/50
    loss: -0.0036 - acc: 0.8873 - KR: 2.8288 - val_loss: -0.0017 - val_acc: 0.8802 - val_KR: 2.8015


.. parsed-literal::

    Epoch 30/50
    loss: -0.0130 - acc: 0.8888 - KR: 2.8301 - val_loss: -0.0019 - val_acc: 0.8792 - val_KR: 2.8037


.. parsed-literal::

    Epoch 31/50
    loss: -0.0001 - acc: 0.8898 - KR: 2.8378 - val_loss: -0.0025 - val_acc: 0.8800 - val_KR: 2.7789


.. parsed-literal::

    Epoch 32/50
    loss: -0.0027 - acc: 0.8893 - KR: 2.8273 - val_loss: -0.0017 - val_acc: 0.8735 - val_KR: 2.8077


.. parsed-literal::

    Epoch 33/50
    loss: 0.0239 - acc: 0.8908 - KR: 2.8385 - val_loss: -0.0013 - val_acc: 0.8770 - val_KR: 2.8136


.. parsed-literal::

    Epoch 34/50
    loss: -0.0139 - acc: 0.8910 - KR: 2.8461 - val_loss: -0.0029 - val_acc: 0.8792 - val_KR: 2.8236


.. parsed-literal::

    Epoch 35/50
    loss: -0.0040 - acc: 0.8901 - KR: 2.8543 - val_loss: -0.0013 - val_acc: 0.8740 - val_KR: 2.8225


.. parsed-literal::

    Epoch 36/50
    loss: -0.0020 - acc: 0.8919 - KR: 2.8619 - val_loss: -0.0025 - val_acc: 0.8800 - val_KR: 2.8071


.. parsed-literal::

    Epoch 37/50
    loss: -0.0067 - acc: 0.8925 - KR: 2.8522 - val_loss: -0.0032 - val_acc: 0.8812 - val_KR: 2.8336


.. parsed-literal::

    Epoch 38/50
    loss: -0.0063 - acc: 0.8916 - KR: 2.8582 - val_loss: -0.0036 - val_acc: 0.8812 - val_KR: 2.8604


.. parsed-literal::

    Epoch 39/50
    loss: -0.0087 - acc: 0.8927 - KR: 2.8672 - val_loss: -0.0033 - val_acc: 0.8846 - val_KR: 2.8692


.. parsed-literal::

    Epoch 40/50
    loss: -0.0147 - acc: 0.8942 - KR: 2.8641 - val_loss: -0.0014 - val_acc: 0.8832 - val_KR: 2.8150


.. parsed-literal::

    Epoch 41/50
    loss: 0.0033 - acc: 0.8928 - KR: 2.8696 - val_loss: -0.0033 - val_acc: 0.8830 - val_KR: 2.8585


.. parsed-literal::

    Epoch 42/50
    loss: -0.0066 - acc: 0.8934 - KR: 2.8735 - val_loss: -0.0030 - val_acc: 0.8809 - val_KR: 2.8260


.. parsed-literal::

    Epoch 43/50
    loss: -0.0146 - acc: 0.8952 - KR: 2.8766 - val_loss: -0.0031 - val_acc: 0.8852 - val_KR: 2.8403


.. parsed-literal::

    Epoch 44/50
    loss: -0.0086 - acc: 0.8950 - KR: 2.8773 - val_loss: -0.0018 - val_acc: 0.8787 - val_KR: 2.9115


.. parsed-literal::

    Epoch 45/50
    loss: -0.0000 - acc: 0.8957 - KR: 2.8799 - val_loss: -0.0040 - val_acc: 0.8863 - val_KR: 2.8622


.. parsed-literal::

    Epoch 46/50
    loss: -0.0104 - acc: 0.8961 - KR: 2.8910 - val_loss: -0.0038 - val_acc: 0.8843 - val_KR: 2.8445


.. parsed-literal::

    Epoch 47/50
    loss: -0.0022 - acc: 0.8953 - KR: 2.8878 - val_loss: -0.0036 - val_acc: 0.8823 - val_KR: 2.8444


.. parsed-literal::

    Epoch 48/50
    loss: -0.0157 - acc: 0.8951 - KR: 2.8893 - val_loss: -0.0044 - val_acc: 0.8867 - val_KR: 2.8650


.. parsed-literal::

    Epoch 49/50
    loss: -0.0080 - acc: 0.8945 - KR: 2.8897 - val_loss: -0.0042 - val_acc: 0.8851 - val_KR: 2.8629


.. parsed-literal::

    Epoch 50/50
    loss: -0.0060 - acc: 0.8966 - KR: 2.8937 - val_loss: -0.0038 - val_acc: 0.8845 - val_KR: 2.8673


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
      (2): ScaledL2NormPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (4): GroupSort2()
      (5): ScaledL2NormPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
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
    Image 0        0.246                1.04
    Image 1        1.863                4.57
    Image 2        0.475                1.78
    Image 3        0.601                2.71
    Image 4        0.108                0.43
    Image 5        0.214                0.83
    Image 6        0.104                0.45
    Image 7        0.447                1.61
    Image 8        1.564                3.89
    Image 9        0.135                0.59


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

