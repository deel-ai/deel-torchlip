Example 3: HKR classifier on MNIST dataset
==========================================

This notebook demonstrates how to learn a binary classifier on the
MNIST0-8 dataset (MNIST with only 0 and 8).

1. Data preparation
-------------------

For this task we will select two classes: 0 and 8. Labels are changed to
{-1,1}, wich is compatible with the Hinge term used in the loss.

.. code:: ipython3

    import torch
    
    from torchvision import datasets
    
    # first we select the two classes
    selected_classes = [0, 8]  # must be two classes as we perform binary classification
    
    
    def prepare_data(dataset, class_a=0, class_b=8):
        """
        This function convert the MNIST data to make it suitable for our binary classification
        setup.
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
    
    # prepare the data
    train = prepare_data(train, selected_classes[0], selected_classes[1])
    test = prepare_data(test, selected_classes[0], selected_classes[1])
    
    # display infos about dataset
    print(
        "train set size: %i samples, classes proportions: %.3f percent"
        % (len(train), 100 * (train.tensors[1] == 1).sum() / len(train))
    )
    print(
        "test set size: %i samples, classes proportions: %.3f percent"
        % (len(test), 100 * (test.tensors[1] == 1).sum() / len(test))
    )


.. parsed-literal::

    train set size: 11774 samples, classes proportions: 50.306 percent
    test set size: 1954 samples, classes proportions: 50.154 percent


2. Build lipschitz Model
------------------------

Here the experiments are done with a only fully-connected layers but
``torchlip`` also provides state-of-the-art 1-Lipschitz convolutional
layers.

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
    from tqdm.notebook import trange, tqdm
    
    # training parameters
    epochs = 5
    batch_size = 128
    
    # loss parameters
    min_margin = 1
    alpha = 10
    
    optimizer = torch.optim.Adam(lr=0.01, params=wass.parameters())
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    
    tepochs = trange(epochs)
    for _ in tepochs:
        m_nc, m_kr, m_hm, m_acc = 0, 0, 0, 0
    
        tdata = tqdm(train_loader)
        wass.train()
        for data, target in tdata:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = wass(data)
            loss = hkr_loss(output, target, alpha=alpha, min_margin=min_margin)
            loss.backward()
            optimizer.step()
    
            m_nc += 1
            m_kr += kr_loss(output, target, (-1, 1))
            m_hm += hinge_margin_loss(output, target, min_margin)
            m_acc += (torch.sign(output).flatten() == torch.sign(target)).sum() / len(
                target
            )
            tdata.set_postfix(
                {
                    k: "{:.04f}".format(v)
                    for k, v in {
                        "loss": loss,
                        "kr": m_kr / m_nc,
                        "hm": m_hm / m_nc,
                        "acc": m_acc / m_nc,
                    }.items()
                }
            )
    
        wass.eval()
        testo = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            testo.append(wass(data).detach().cpu())
        testo = torch.cat(testo).flatten()
    
        postfix = {
            f"train_{k}": "{:.04f}".format(v)
            for k, v in {
                "loss": loss,
                "kr": m_kr / m_nc,
                "hm": m_hm / m_nc,
                "acc": m_acc / m_nc,
            }.items()
        }
        postfix.update(
            {
                f"val_{k}": "{:.04f}".format(v)
                for k, v in {
                    "loss": hkr_loss(
                        testo, test.tensors[1], alpha=alpha, min_margin=min_margin
                    ),
                    "kr": kr_loss(testo.flatten(), test.tensors[1], (-1, 1)),
                    "hm": hinge_margin_loss(testo.flatten(), test.tensors[1], min_margin),
                    "acc": (torch.sign(testo).flatten() == torch.sign(test.tensors[1]))
                    .float()
                    .mean(),
                }.items()
            }
        )
    
        tepochs.set_postfix(postfix)



.. parsed-literal::

      0%|          | 0/5 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/92 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/92 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/92 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/92 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/92 [00:00<?, ?it/s]


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

    tensor(0.3460)


.. code:: ipython3

    from scipy.spatial.distance import pdist
    
    wass.eval()
    
    p = []
    for batch, _ in tqdm(train_loader):
        x = batch.numpy()
        y = wass(batch.to(device)).detach().cpu().numpy()
        xd = pdist(x.reshape(batch.shape[0], -1))
        yd = pdist(y.reshape(batch.shape[0], -1))
    
        p.append((yd / xd).max())
    print(torch.tensor(p).max())



.. parsed-literal::

      0%|          | 0/92 [00:00<?, ?it/s]


.. parsed-literal::

    tensor(0.9074, dtype=torch.float64)


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
    SpectralLinear(in_features=784, out_features=128, bias=True), min=0.9997813701629639, max=1.0002498626708984
    SpectralLinear(in_features=128, out_features=64, bias=True), min=0.9998164176940918, max=1.0001946687698364
    SpectralLinear(in_features=64, out_features=32, bias=True), min=0.99966961145401, max=1.0003477334976196
    FrobeniusLinear(in_features=32, out_features=1, bias=True), min=1.0, max=1.0


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
    Linear(in_features=784, out_features=128, bias=True), min=0.9997813701629639, max=1.0002498626708984
    Linear(in_features=128, out_features=64, bias=True), min=0.9998164176940918, max=1.0001946687698364
    Linear(in_features=64, out_features=32, bias=True), min=0.99966961145401, max=1.0003477334976196
    Linear(in_features=32, out_features=1, bias=True), min=1.0, max=1.0


As we can see, all our singular values are very close to one.

.. container:: alert alert-block alert-danger