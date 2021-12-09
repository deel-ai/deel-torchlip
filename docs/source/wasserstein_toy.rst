Example 1: Wasserstein distance estimation
==========================================

In this notebook we estimate the wasserstein distance through its
Kantorovich-Rubinestein dual representation by using a 1-Lipschitz
neural network by using.

1. Wasserstein distance
-----------------------

The wasserstein distance measure the distance between two probability
distributions. The Wikipedia article gives a more intuitive definition
of it:

   Intuitively, if each distribution is viewed as a unit amount of
   “dirt” piled on :math:`M`, the metric is the minimum “cost” of
   turning one pile into the other, which is assumed to be the amount of
   dirt that needs to be moved times the mean distance it has to be
   moved. Because of this analogy, the metric is known in computer
   science as the earth mover’s distance.

Mathematically it is defined as

.. math::


   W_1(\mu,\nu) = \inf_{\pi \in \Pi(\mu,\nu)}\underset{x,z \sim \pi}{\mathbb{E}}\Vert{} \textbf{x}-\textbf{z} \Vert{}

where :math:`\Pi(\mu,\nu)` is the set of all probability measures on
:math:`\Omega\times \Omega` with marginals :math:`\mu` and :math:`\nu`.
In most case this equation is not tractable.

2. Parameters input images
--------------------------

We illustrate this on a synthetic image dataset where the :math:`W_1`
distance is known.

Our synthetic dataset contains images with black or white squares,
allowing us to check if the computed wasserstein distance is correct.
The two distributions are

-  the set of black images (all 0),
-  the set of images with a square on it (all 0, with a square of -1 or
   +1 in the middle).

.. code:: ipython3

    from typing import Tuple

    import matplotlib.pyplot as plt
    import numpy as np

    size = (64, 64)
    frac = 0.3  # proportion of the center square


    def generate_toy_images(shape: Tuple[int, int], frac: float = 0, value: float = 1):
        """
        Function that generate a single image.

        Args:
            shape: Shape of the output image.
            frac: Proportion of the center rectangle.
            value: Value assigned to the center rectangle.
        """
        img = np.zeros(shape)
        if frac == 0:
            return img

        frac = frac ** 0.5

        l = int(shape[0] * frac)
        ldec = (shape[0] - l) // 2
        w = int(shape[1] * frac)
        wdec = (shape[1] - w) // 2

        img[ldec : ldec + l, wdec : wdec + w] = value

        return img


    def generator(batch_size: int, shape: Tuple[int, int], frac: float):
        """
        Creates an infinite generator that generates batch of images. Half of the batch
        comes from the first distribution (only black images), while the remaining half
        comes from the second distribution.

        Args:
            batch_size: Number of images in each batch.
            shape: Shape of the image.
            frac: Fraction of the square to set "white".

        Returns:
            An infinite generator that yield batch of the given size.
        """

        pwhite = generate_toy_images(shape, frac=frac, value=1)
        nwhite = generate_toy_images(shape, frac=frac, value=-1)

        nblack = batch_size // 2
        nsquares = batch_size - nblack
        npwhite = nsquares // 2
        nnwhite = nsquares - npwhite

        batch_x = np.concatenate(
            (
                np.zeros((nblack,) + shape),
                np.repeat(pwhite[None, ...], npwhite, axis=0),
                np.repeat(nwhite[None, ...], nnwhite, axis=0),
            ),
            axis=0,
        )
        batch_y = np.concatenate((np.zeros((nblack, 1)), np.ones((nsquares, 1))), axis=0)

        while True:
            yield batch_x, batch_y


    def display_image(ax, image, title: str = ""):
        """
        Small function to display images.
        """
        ax.imshow(image, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)


.. parsed-literal::

    Matplotlib created a temporary config/cache directory at /tmp/matplotlib-rxejzzl3 because the default path (/home/justin.plakoo/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.


We consider images of size 64x64, and an inner square that covers about
30% of the image. We can manually compute the :math:`W_1` distance
between the two sets.

.. code:: ipython3

    img1 = generate_toy_images(size, 0)
    img2 = generate_toy_images(size, frac, value=-1)
    img3 = generate_toy_images(size, frac, value=1)

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    display_image(axs[0], img1, "black (label = -1)")
    display_image(axs[1], img2, "'negative' white (label = 1)")
    display_image(axs[2], img3, "'positive' white (label = 1)")

    print("L2-Norm, black vs. 'negative' white -> {}".format(np.linalg.norm(img2 - img1)))
    print("L2-Norm, black vs. 'positive' white -> {}".format(np.linalg.norm(img3 - img1)))


.. parsed-literal::

    L2-Norm, black vs. 'negative' white -> 35.0
    L2-Norm, black vs. 'positive' white -> 35.0



.. image:: wasserstein_toy_files/wasserstein_toy_4_1.png


As we can see, the distance between the fully black image and any of the
two images with an inner square is :math:`35`, and see these are the
only images in our distributions, the :math:`W_1` distance between the
two distances is also :math:`35`.

3. Kantorovich-Rubinestein dual formulation
-------------------------------------------

The Kantorovich-Rubinestein (KR) dual formulation of the Wasserstein
distance is

.. math::  W_1(\mu, \nu) = \sup_{f \in Lip_1(\Omega)} \underset{\textbf{x} \sim \mu}{\mathbb{E}} \left[f(\textbf{x} )\right] -\underset{\textbf{x}  \sim \nu}{\mathbb{E}} \left[f(\textbf{x} )\right].

This state the problem as an optimization problem over the space of
1-Lipschitz functions. We can estimate this by optimizing over the space
of 1-Lipschitz neural networks.

-  [1] C. Anil, J. Lucas, et R. Grosse, “Sorting out Lipschitz function
   approximation”, arXiv:1811.05381, nov. 2018.

3.1. Building lipschitz Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we use the ``deel.torchlip`` (short ``torchlip``) to
build a 1-Lipschitz network. The ``torchlip`` library is the PyTorch
equivalent of ```deel-lip`` <https://github.com/deel-ai/deel-lip>`__. In
this example, we use two 1-Lipschitz layers and a special activation
function:

-  ``SpectralLinear`` uses spectral normalization to force the maximum
   singular value of the weight matrix to be one, followed by Bjorck
   normalization to force all singular values to be 1. After
   convergence, all singular values are equal to 1 and the linear
   operation is 1-Lipschitz. The ``SpectralLinear`` class also uses
   orthogonal initialization for the weight (see
   ``torch.init.orthogonal_``).
-  ``FrobeniusLinear`` simply divide the weight matrix by its Frobenius
   norm. We only use it for the last layer because this layer has a
   single output. Similar to ``SpectralLinear``, the weights are
   initialized using orthogonal initialization.
-  We use ``FullSort`` activation, which is a 1-Lipschitz activation.

.. code:: ipython3

    import torch
    from deel import torchlip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wass = torchlip.Sequential(
        torch.nn.Flatten(),
        torchlip.SpectralLinear(np.prod(size), 128),
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
      (1): SpectralLinear(in_features=4096, out_features=128, bias=True)
      (2): FullSort()
      (3): SpectralLinear(in_features=128, out_features=64, bias=True)
      (4): FullSort()
      (5): SpectralLinear(in_features=64, out_features=32, bias=True)
      (6): FullSort()
      (7): FrobeniusLinear(in_features=32, out_features=1, bias=True)
    )



3.2. Training a 1-Lipschitz network with KR loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now train this neural network using the Kantorovich-Rubinestein
formulation for the Wasserstein distance.

.. code:: ipython3

    from deel.torchlip.functional import kr_loss
    from tqdm.notebook import trange, tqdm

    batch_size = 16
    n_epochs = 10
    steps_per_epoch = 256

    # Create the image generator:
    g = generator(batch_size, size, frac)

    optimizer = torch.optim.Adam(lr=0.01, params=wass.parameters())

    n_steps = steps_per_epoch // batch_size
    tepochs = trange(n_epochs)
    tsteps = trange(n_steps)

    for epoch in tepochs:
        tsteps.reset()
        for _ in range(n_steps):
            data, target = next(g)
            data, target = (
                torch.tensor(data).float().to(device),
                torch.tensor(target).float().to(device),
            )
            optimizer.zero_grad()
            output = wass(data)
            loss = kr_loss(output, target)
            loss.backward()
            optimizer.step()
            tsteps.set_postfix({"loss": "{:.6f}".format(loss)})
            tsteps.update()
        tsteps.refresh()
        tepochs.set_postfix({"loss": "{:.6f}".format(loss)})



.. parsed-literal::

      0%|          | 0/10 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/16 [00:00<?, ?it/s]


As we can see the loss converge to the value :math:`35` which is the
:math:`W_1` distance between the two distributions (with and without
squares).

.. container:: alert alert-block alert-danger
