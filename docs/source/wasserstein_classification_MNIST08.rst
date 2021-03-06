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

    Sequential model contains a layer which is not a Lipschitz layer: Flatten(start_dim=1, end_dim=-1)




.. parsed-literal::

    Sequential(
      (0): Flatten(start_dim=1, end_dim=-1)
      (1): SpectralLinear(in_features=784, out_features=128, bias=True)
      (2): FullSort()
      (3): SpectralLinear(in_features=128, out_features=64, bias=True)
      (4): FullSort()
      (5): SpectralLinear(in_features=64, out_features=32, bias=True)
      (6): FullSort()
      (7): FrobeniusLinear(in_features=32, out_features=1, bias=True)
    )



3. Learn classification on MNIST
--------------------------------

.. code:: ipython3

    from deel.torchlip.functional import kr_loss, hkr_loss, hinge_margin_loss

    # training parameters
    epochs = 10
    batch_size = 128

    # loss parameters
    min_margin = 1
    alpha = 10

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
            loss = hkr_loss(output, target, alpha=alpha, min_margin=min_margin)
            loss.backward()
            optimizer.step()

            # Compute metrics on batch
            m_kr += kr_loss(output, target, (1, -1))
            m_hm += hinge_margin_loss(output, target, min_margin)
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
                    testo, test.tensors[1], alpha=alpha, min_margin=min_margin
                ),
                "KR": kr_loss(testo.flatten(), test.tensors[1], (1, -1)),
                "acc": (torch.sign(testo).flatten() == torch.sign(test.tensors[1]))
                .float()
                .mean(),
            }.items()
        ]

        print(f"Epoch {epoch + 1}/{epochs}")
        print(" - ".join(metrics))



.. parsed-literal::

    Epoch 1/10
    loss: -2.5269 - KR: 1.6177 - acc: 0.8516 - val_loss: -2.7241 - val_KR: 3.0157 - val_acc: 0.9939
    Epoch 2/10
    loss: -3.6040 - KR: 3.8627 - acc: 0.9918 - val_loss: -4.5285 - val_KR: 4.7897 - val_acc: 0.9918
    Epoch 3/10
    loss: -5.7646 - KR: 5.4015 - acc: 0.9922 - val_loss: -5.7246 - val_KR: 6.0067 - val_acc: 0.9898
    Epoch 4/10
    loss: -6.6268 - KR: 6.2105 - acc: 0.9921 - val_loss: -6.2183 - val_KR: 6.4874 - val_acc: 0.9893
    Epoch 5/10
    loss: -6.4072 - KR: 6.5715 - acc: 0.9931 - val_loss: -6.4530 - val_KR: 6.7446 - val_acc: 0.9887
    Epoch 6/10
    loss: -6.7689 - KR: 6.7803 - acc: 0.9926 - val_loss: -6.6342 - val_KR: 6.8849 - val_acc: 0.9898
    Epoch 7/10
    loss: -6.2389 - KR: 6.8948 - acc: 0.9932 - val_loss: -6.7603 - val_KR: 6.9643 - val_acc: 0.9933
    Epoch 8/10
    loss: -6.9207 - KR: 6.9642 - acc: 0.9933 - val_loss: -6.8199 - val_KR: 7.0147 - val_acc: 0.9918
    Epoch 9/10
    loss: -6.9446 - KR: 7.0211 - acc: 0.9936 - val_loss: -6.8038 - val_KR: 7.0666 - val_acc: 0.9887
    Epoch 10/10
    loss: -6.5403 - KR: 7.0694 - acc: 0.9942 - val_loss: -6.9136 - val_KR: 7.1086 - val_acc: 0.9933


4. Evaluate the Lipschitz constant of our networks
--------------------------------------------------

4.1. Empirical evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

We can estimate the Lipschitz constant by evaluating

.. math::


       \frac{\Vert{}F(x_2) - F(x_1)\Vert{}}{\Vert{}x_2 - x_1\Vert{}} \quad\text{or}\quad
       \frac{\Vert{}F(x + \epsilon) - F(x)\Vert{}}{\Vert{}\epsilon\Vert{}}

for various inputs.

.. code:: ipython3

    from scipy.spatial.distance import pdist

    wass.eval()

    p = []
    for _ in range(64):
        eps = 1e-3
        batch, _ = next(iter(train_loader))
        dist = torch.distributions.Uniform(-eps, +eps).sample(batch.shape)
        y1 = wass(batch.to(device)).detach().cpu()
        y2 = wass((batch + dist).to(device)).detach().cpu()

        p.append(
            torch.max(
                torch.norm(y2 - y1, dim=1)
                / torch.norm(dist.reshape(dist.shape[0], -1), dim=1)
            )
        )
    print(torch.tensor(p).max())


.. parsed-literal::

    tensor(0.1349)


.. code:: ipython3

    p = []
    for batch, _ in train_loader:
        x = batch.numpy()
        y = wass(batch.to(device)).detach().cpu().numpy()
        xd = pdist(x.reshape(batch.shape[0], -1))
        yd = pdist(y.reshape(batch.shape[0], -1))

        p.append((yd / xd).max())
    print(torch.tensor(p).max())


.. parsed-literal::

    tensor(0.9038, dtype=torch.float64)


As we can see, using the :math:`\epsilon`-version, we greatly
under-estimate the Lipschitz constant. Using the train dataset, we find
a Lipschitz constant close to 0.9, which is better, but our network
should be 1-Lipschitz.

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
            print(f"{layer}, min={s.min()}, max={s.max()}")


.. parsed-literal::

    === Before export ===
    SpectralLinear(in_features=784, out_features=128, bias=True), min=0.9999998807907104, max=1.0
    SpectralLinear(in_features=128, out_features=64, bias=True), min=0.9999998807907104, max=1.0000001192092896
    SpectralLinear(in_features=64, out_features=32, bias=True), min=0.9999998807907104, max=1.0
    FrobeniusLinear(in_features=32, out_features=1, bias=True), min=0.9999999403953552, max=0.9999999403953552


.. code:: ipython3

    wexport = wass.vanilla_export()

    print("=== After export ===")
    layers = list(wexport.children())
    for layer in layers:
        if hasattr(layer, "weight"):
            w = layer.weight
            u, s, v = torch.svd(w)
            print(f"{layer}, min={s.min()}, max={s.max()}")


.. parsed-literal::

    === After export ===
    Linear(in_features=784, out_features=128, bias=True), min=0.9999998807907104, max=1.0
    Linear(in_features=128, out_features=64, bias=True), min=0.9999998807907104, max=1.0000001192092896
    Linear(in_features=64, out_features=32, bias=True), min=0.9999998807907104, max=1.0
    Linear(in_features=32, out_features=1, bias=True), min=0.9999999403953552, max=0.9999999403953552


As we can see, all our singular values are very close to one.

.. container:: alert alert-block alert-danger
