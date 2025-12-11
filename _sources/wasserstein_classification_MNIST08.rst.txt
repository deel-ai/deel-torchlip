Example 3: HKR classifier on MNIST dataset
==========================================

|Open In Colab|

This notebook demonstrates how to learn a binary classifier on the
MNIST0-8 dataset (MNIST with only 0 and 8).

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/deel-ai/deel-torchlip/blob/master/docs/notebooks/wasserstein_classification_MNIST08.ipynb

.. code:: ipython3

    # Install the required library deel-torchlip (uncomment line below)
    # %pip install -qqq deel-torchlip

1. Data preparation
-------------------

For this task we will select two classes: 0 and 8. Labels are changed to
{-1,1}, which is compatible with the hinge term used in the loss.

.. code:: ipython3

    import torch
    from torchvision import datasets
    
    # First we select the two classes
    selected_classes = [0, 8]  # must be two classes as we perform binary classification
    
    
    def prepare_data(dataset, class_a=0, class_b=8):
        """
        This function converts the MNIST data to make it suitable for our binary
        classification setup.
        """
        x = dataset.data
        y = dataset.targets
        # select items from the two selected classes
        mask = (y == class_a) + (
            y == class_b
        )  # mask to select only items from class_a or class_b
        x = x[mask]
        y = y[mask]
    
        # convert from range int[0,255] to float32[-1,1]
        x = x.float() / 255
        x = x.reshape((-1, 28, 28, 1))
        # change label to binary classification {-1,1}
    
        y_ = torch.zeros_like(y).float()
        y_[y == class_a] = 1.0
        y_[y == class_b] = -1.0
        return torch.utils.data.TensorDataset(x, y_)
    
    
    train = datasets.MNIST("./data", train=True, download=True)
    test = datasets.MNIST("./data", train=False, download=True)
    
    # Prepare the data
    train = prepare_data(train, selected_classes[0], selected_classes[1])
    test = prepare_data(test, selected_classes[0], selected_classes[1])
    
    # Display infos about dataset
    print(
        f"Train set size: {len(train)} samples, classes proportions: "
        f"{100 * (train.tensors[1] == 1).numpy().mean():.2f} %"
    )
    print(
        f"Test set size: {len(test)} samples, classes proportions: "
        f"{100 * (test.tensors[1] == 1).numpy().mean():.2f} %"
    )
    



.. parsed-literal::

    Train set size: 11774 samples, classes proportions: 50.31 %
    Test set size: 1954 samples, classes proportions: 50.15 %


2. Build Lipschitz model
------------------------

Here, the experiments are done with a model with only fully-connected
layers. However, ``torchlip`` also provides state-of-the-art 1-Lipschitz
convolutional layers.

.. code:: ipython3

    import torch
    from deel import torchlip
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ninputs = 28 * 28
    wass = torchlip.Sequential(
        torch.nn.Flatten(),
        torchlip.SpectralLinear(ninputs, 128),
        torchlip.FullSort(),
        torchlip.SpectralLinear(128, 64),
        torchlip.FullSort(),
        torchlip.SpectralLinear(64, 32),
        torchlip.FullSort(),
        torchlip.FrobeniusLinear(32, 1),
    ).to(device)
    
    wass





.. parsed-literal::

    Sequential(
      (0): Flatten(start_dim=1, end_dim=-1)
      (1): ParametrizedSpectralLinear(
        in_features=784, out_features=128, bias=True
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _SpectralNorm()
            (1): _BjorckNorm()
          )
        )
      )
      (2): FullSort()
      (3): ParametrizedSpectralLinear(
        in_features=128, out_features=64, bias=True
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _SpectralNorm()
            (1): _BjorckNorm()
          )
        )
      )
      (4): FullSort()
      (5): ParametrizedSpectralLinear(
        in_features=64, out_features=32, bias=True
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _SpectralNorm()
            (1): _BjorckNorm()
          )
        )
      )
      (6): FullSort()
      (7): ParametrizedFrobeniusLinear(
        in_features=32, out_features=1, bias=True
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _FrobeniusNorm()
          )
        )
      )
    )



3. Learn classification on MNIST
--------------------------------

.. code:: ipython3

    from deel.torchlip import KRLoss, HKRLoss, HingeMarginLoss
    
    # training parameters
    epochs = 10
    batch_size = 128
    
    # loss parameters
    min_margin = 1
    alpha = 0.98
    
    kr_loss = KRLoss()
    hkr_loss = HKRLoss(alpha=alpha, min_margin=min_margin)
    hinge_margin_loss =HingeMarginLoss(min_margin=min_margin)
    optimizer = torch.optim.Adam(lr=0.001, params=wass.parameters())
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    
    for epoch in range(epochs):
    
        m_kr, m_hm, m_acc = 0, 0, 0
        wass.train()
    
        for step, (data, target) in enumerate(train_loader):
    
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = wass(data)
            loss = hkr_loss(output, target)
            loss.backward()
            optimizer.step()
    
            # Compute metrics on batch
            m_kr += kr_loss(output, target)
            m_hm += hinge_margin_loss(output, target)
            m_acc += (torch.sign(output).flatten() == torch.sign(target)).sum() / len(
                target
            )
    
        # Train metrics for the current epoch
        metrics = [
            f"{k}: {v:.04f}"
            for k, v in {
                "loss": loss,
                "KR": m_kr / (step + 1),
                "acc": m_acc / (step + 1),
            }.items()
        ]
    
        # Compute test loss for the current epoch
        wass.eval()
        testo = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            testo.append(wass(data).detach().cpu())
        testo = torch.cat(testo).flatten()
    
        # Validation metrics for the current epoch
        metrics += [
            f"val_{k}: {v:.04f}"
            for k, v in {
                "loss": hkr_loss(
                    testo, test.tensors[1]
                ),
                "KR": kr_loss(testo.flatten(), test.tensors[1]),
                "acc": (torch.sign(testo).flatten() == torch.sign(test.tensors[1]))
                .float()
                .mean(),
            }.items()
        ]
    
        print(f"Epoch {epoch + 1}/{epochs}")
        print(" - ".join(metrics))



.. parsed-literal::

    Epoch 1/10
    loss: -0.0341 - KR: 1.2932 - acc: 0.8849 - val_loss: -0.0332 - val_KR: 2.2678 - val_acc: 0.9918


.. parsed-literal::

    Epoch 2/10
    loss: -0.0578 - KR: 2.7714 - acc: 0.9918 - val_loss: -0.0588 - val_KR: 3.3804 - val_acc: 0.9928


.. parsed-literal::

    Epoch 3/10
    loss: -0.0787 - KR: 3.8889 - acc: 0.9930 - val_loss: -0.0777 - val_KR: 4.5303 - val_acc: 0.9903


.. parsed-literal::

    Epoch 4/10
    loss: -0.0989 - KR: 4.8591 - acc: 0.9946 - val_loss: -0.0949 - val_KR: 5.2321 - val_acc: 0.9928


.. parsed-literal::

    Epoch 5/10
    loss: -0.1063 - KR: 5.4602 - acc: 0.9948 - val_loss: -0.1064 - val_KR: 5.7756 - val_acc: 0.9923


.. parsed-literal::

    Epoch 6/10
    loss: -0.1025 - KR: 5.8678 - acc: 0.9946 - val_loss: -0.1123 - val_KR: 6.0558 - val_acc: 0.9928


.. parsed-literal::

    Epoch 7/10
    loss: -0.1155 - KR: 6.0850 - acc: 0.9958 - val_loss: -0.1178 - val_KR: 6.2237 - val_acc: 0.9939


.. parsed-literal::

    Epoch 8/10
    loss: -0.0989 - KR: 6.2505 - acc: 0.9957 - val_loss: -0.1193 - val_KR: 6.3435 - val_acc: 0.9933


.. parsed-literal::

    Epoch 9/10
    loss: -0.1330 - KR: 6.3675 - acc: 0.9956 - val_loss: -0.1218 - val_KR: 6.4925 - val_acc: 0.9939


.. parsed-literal::

    Epoch 10/10
    loss: -0.1272 - KR: 6.4819 - acc: 0.9966 - val_loss: -0.1148 - val_KR: 6.5502 - val_acc: 0.9862


4. Evaluate the Lipschitz constant of our networks
--------------------------------------------------

4.1. Empirical evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

We can estimate the Lipschitz constant by evaluating

.. math::


       \frac{\Vert{}F(x_2) - F(x_1)\Vert{}}{\Vert{}x_2 - x_1\Vert{}} \quad\text{or}\quad 
       \frac{\Vert{}F(x + \epsilon) - F(x)\Vert{}}{\Vert{}\epsilon\Vert{}}

for various inputs.

The deel.torchlip.utils.evaluate_lip_const implements several methods to
evaluate this constant, either by adding random noise :math:`x+\epsilon`
and evaluating
:math:`\frac{\Vert{}F(x + \epsilon) - F(x)\Vert{}}{\Vert{}\epsilon\Vert{}}`,
or by an adversarial attack on :math:`\epsilon` to increase this value,
or by computing the jacobian norm :math:`||\nabla_x F(x)||`

.. code:: ipython3

    from deel.torchlip.utils import evaluate_lip_const

.. code:: ipython3

    x,y = next(iter(test_loader))
    evaluate_lip_const(wass, x.to(device), evaluation_type="all", disjoint_neurons=False, double_attack=True)


.. parsed-literal::

    Empirical lipschitz constant is 1.0000001192092896 with method jacobian_norm
    Empirical lipschitz constant is 0.16324962675571442 with method noise_norm
    Warning : double_attack is set to True,                 the computation time will be doubled


.. parsed-literal::

    Empirical lipschitz constant is 0.9863337874412537 with method attack




.. parsed-literal::

    1.0000001192092896



As we can see, each method provide an underestimation of the Lipschitz
constant. The adversarial attack method and jacobian norm find values
very close to one.

4.1. Singular-Value Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since our network is only made of linear layers and ``FullSort``
activation, we can compute *Singular-Value Decomposition* (SVD) of our
weight matrix and check that, for each linear layer, all singular values
are 1.

.. code:: ipython3

    print("=== Before export ===")
    layers = list(wass.children())
    for layer in layers:
        if hasattr(layer, "weight"):
            w = layer.weight
            u, s, v = torch.svd(w)
            print(f"{type(layer)}, min={s.min()}, max={s.max()}")


.. parsed-literal::

    === Before export ===
    <class 'abc.ParametrizedSpectralLinear'>, min=0.9999998807907104, max=1.0
    <class 'abc.ParametrizedSpectralLinear'>, min=1.000000238418579, max=1.000009536743164
    <class 'abc.ParametrizedSpectralLinear'>, min=0.9999998807907104, max=1.0
    <class 'abc.ParametrizedFrobeniusLinear'>, min=0.9999998211860657, max=0.9999998211860657


4.2 Model export
~~~~~~~~~~~~~~~~

Once training is finished, the model can be optimized for inference by
using the ``vanilla_export()`` method. The ``torchlip`` layers are
converted to their PyTorch counterparts, e.g. ``SpectralConv2d``
layers will be converted into ``torch.nn.Conv2d`` layers.

Warnings:
^^^^^^^^^

vanilla_export method modifies the model in-place.

In order to build and export a new model while keeping the reference
one, it is required to follow these steps:

# Build e new mode for instance with torchlip.Sequential(
torchlip.SpectralConv2d(…), …)

``wexport = <your_function_to_build_the_model>()``

# Copy the parameters from the reference t the new model

``wexport.load_state_dict(wass.state_dict())``

# one forward required to initialize pamatrizations

``vanilla_model(one_input)``

# vanilla_export the new model

``wexport = wexport.vanilla_export()``

.. code:: ipython3

    wexport = wass.vanilla_export()
    
    print("=== After export ===")
    layers = list(wexport.children())
    for layer in layers:
        if hasattr(layer, "weight"):
            w = layer.weight
            u, s, v = torch.svd(w)
            print(f"{type(layer)}, min={s.min()}, max={s.max()}")


.. parsed-literal::

    === After export ===
    <class 'torch.nn.modules.linear.Linear'>, min=0.9999998807907104, max=1.0
    <class 'torch.nn.modules.linear.Linear'>, min=1.000000238418579, max=1.000009536743164
    <class 'torch.nn.modules.linear.Linear'>, min=0.9999998807907104, max=1.0
    <class 'torch.nn.modules.linear.Linear'>, min=0.9999998211860657, max=0.9999998211860657


As we can see, all our singular values are very close to one.

.. container:: alert alert-block alert-danger
