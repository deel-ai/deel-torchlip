Example 4: HKR multiclass and fooling
=====================================

This notebook will show how to train a Lispchitz network in a multiclass
configuration. The HKR (hinge-Kantorovich-Rubinstein) loss is extended
to multiclass using a one-vs all setup. The notebook will go through the
process of designing and training the network. It will also show how to
compute robustness certificates from the outputs of the network. Finally
the guarantee of these certificates will be checked by attacking the
network.

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

    batch_size = 4096
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size)


2. Model architecture
---------------------

The original one-vs-all setup would require 10 different networks (1 per
class). However, we use in practice a network with a common body and a
Lipschitz head (linear layer) containing 10 output neurons, like any
standard network for multiclass classification. Note that each head
neuron is not a 1-Lipschitz function; however the overall head with the
10 outputs is 1-Lipschitz.

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
        torchlip.SpectralLinear(64, 10, bias=False),
        # Similarly, model has a parameter to set the Lipschitz constant that automatically
        # sets the constant of each layer.
        k_coef_lip=1.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



.. parsed-literal::

    Sequential model contains a layer which is not a Lipschitz layer: Flatten(start_dim=1, end_dim=-1)




.. parsed-literal::

    Sequential(
      (0): SpectralConv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (1): GroupSort2()
      (2): ScaledL2NormPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
      (3): SpectralConv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (4): GroupSort2()
      (5): ScaledL2NormPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
      (6): Flatten(start_dim=1, end_dim=-1)
      (7): SpectralLinear(in_features=1568, out_features=64, bias=True)
      (8): GroupSort2()
      (9): SpectralLinear(in_features=64, out_features=10, bias=False)
    )



3. HKR loss and training
------------------------

The multiclass HKR loss can be found in the ``hkr_multiclass_loss``
function or in the ``HKRMulticlassLoss`` class. The loss has two
parameters: ``alpha`` and ``min_margin``. Decreasing ``alpha`` and
increasing ``min_margin`` improve robustness (at the cost of accuracy).
Note also in the case of Lipschitz networks, more robustness requires
more parameters. For more information, see `our
paper <https://arxiv.org/abs/2006.06520>`__.

In this setup, choosing ``alpha=100`` and ``min_margin=.25`` provides
good robustness without hurting the accuracy too much.

Finally the ``kr_multiclass_loss`` gives an indication on the robustness
of the network (proxy of the average certificate).

.. code:: ipython3

    from tqdm import tqdm

    epochs = 100
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
    hkr_loss = torchlip.HKRMulticlassLoss(alpha=100, min_margin=0.25)

    for epoch in range(epochs):
        m_kr, m_acc = 0, 0

        print(f"Epoch {epoch + 1}/{epochs}")
        with tqdm(total=len(train_loader)) as tsteps:
            for step, (data, target) in enumerate(train_loader):
                tsteps.update()

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
                m_kr += torchlip.functional.kr_multiclass_loss(output, target)
                m_acc += (output.argmax(dim=1) == target.argmax(dim=1)).sum() / len(target)

                # Print metrics of current batch
                postfix = {
                    k: "{:.04f}".format(v)
                    for k, v in {
                        "loss": loss,
                        "acc": m_acc / (step + 1),
                        "kr": m_kr / (step + 1),
                    }.items()
                }
                tsteps.set_postfix(postfix)

            # Compute test loss for the current epoch
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
            val_kr = torchlip.functional.kr_multiclass_loss(test_output, test_targets)
            val_acc = (
                (test_output.argmax(dim=1) == test_targets.argmax(dim=1)).float().mean()
            )

            # Print metrics for the current epoch
            postfix.update(
                {
                    f"val_{k}": f"{v:.04f}"
                    for k, v in {
                        "loss": hkr_loss(test_output, test_targets),
                        "acc": (test_output.argmax(dim=1) == test_targets.argmax(dim=1))
                        .float()
                        .mean(),
                        "kr": torchlip.functional.kr_multiclass_loss(
                            test_output, test_targets
                        ),
                    }.items()
                }
            )
            tsteps.set_postfix(postfix)



.. parsed-literal::

    Epoch 1/100


.. parsed-literal::

    100%|██████████████| 15/15 [00:05<00:00,  2.67it/s, loss=30.2000, acc=0.1436, kr=0.0810, val_loss=29.1302, val_acc=0.2763, val_kr=0.1965]


.. parsed-literal::

    Epoch 2/100


.. parsed-literal::

    100%|██████████████| 15/15 [00:05<00:00,  2.67it/s, loss=20.0201, acc=0.4963, kr=0.2746, val_loss=19.6045, val_acc=0.6003, val_kr=0.3541]


.. parsed-literal::

    Epoch 3/100


.. parsed-literal::

    100%|██████████████| 15/15 [00:05<00:00,  2.69it/s, loss=15.6437, acc=0.6324, kr=0.4195, val_loss=15.5583, val_acc=0.6358, val_kr=0.4814]


.. parsed-literal::

    Epoch 4/100


.. parsed-literal::

    100%|██████████████| 15/15 [00:05<00:00,  2.66it/s, loss=13.4755, acc=0.6517, kr=0.5201, val_loss=13.5746, val_acc=0.6592, val_kr=0.5528]


.. parsed-literal::

    Epoch 5/100


.. parsed-literal::

    100%|██████████████| 15/15 [00:05<00:00,  2.68it/s, loss=12.3127, acc=0.6765, kr=0.5865, val_loss=12.3308, val_acc=0.6770, val_kr=0.6138]


.. parsed-literal::

    Epoch 6/100


.. parsed-literal::

    100%|██████████████| 15/15 [00:05<00:00,  2.66it/s, loss=10.9644, acc=0.6992, kr=0.6388, val_loss=11.5213, val_acc=0.6943, val_kr=0.6577]


.. parsed-literal::

    Epoch 7/100


.. parsed-literal::

    100%|██████████████| 15/15 [00:05<00:00,  2.71it/s, loss=10.5638, acc=0.7156, kr=0.6802, val_loss=10.9288, val_acc=0.7090, val_kr=0.6952]


.. parsed-literal::

    Epoch 8/100


.. parsed-literal::

    100%|██████████████| 15/15 [00:05<00:00,  2.76it/s, loss=10.1280, acc=0.7253, kr=0.7155, val_loss=10.4868, val_acc=0.7216, val_kr=0.7281]


.. parsed-literal::

    Epoch 9/100


.. parsed-literal::

    100%|██████████████| 15/15 [00:05<00:00,  2.74it/s, loss=10.0916, acc=0.7356, kr=0.7459, val_loss=10.1050, val_acc=0.7279, val_kr=0.7571]


.. parsed-literal::

    Epoch 10/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=9.5023, acc=0.7417, kr=0.7745, val_loss=9.7742, val_acc=0.7334, val_kr=0.7828]


.. parsed-literal::

    Epoch 11/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=9.0971, acc=0.7470, kr=0.7990, val_loss=9.4973, val_acc=0.7422, val_kr=0.8060]


.. parsed-literal::

    Epoch 12/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=9.0802, acc=0.7522, kr=0.8220, val_loss=9.2430, val_acc=0.7486, val_kr=0.8271]


.. parsed-literal::

    Epoch 13/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.76it/s, loss=9.0801, acc=0.7555, kr=0.8440, val_loss=9.0003, val_acc=0.7522, val_kr=0.8507]


.. parsed-literal::

    Epoch 14/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.73it/s, loss=8.5745, acc=0.7618, kr=0.8657, val_loss=8.7841, val_acc=0.7523, val_kr=0.8709]


.. parsed-literal::

    Epoch 15/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=8.0554, acc=0.7647, kr=0.8858, val_loss=8.5687, val_acc=0.7561, val_kr=0.8914]


.. parsed-literal::

    Epoch 16/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=8.0398, acc=0.7679, kr=0.9071, val_loss=8.3919, val_acc=0.7606, val_kr=0.9097]


.. parsed-literal::

    Epoch 17/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=7.9438, acc=0.7712, kr=0.9251, val_loss=8.2221, val_acc=0.7663, val_kr=0.9294]


.. parsed-literal::

    Epoch 18/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=8.0154, acc=0.7743, kr=0.9449, val_loss=8.0543, val_acc=0.7648, val_kr=0.9494]


.. parsed-literal::

    Epoch 19/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=7.7195, acc=0.7766, kr=0.9647, val_loss=7.8735, val_acc=0.7718, val_kr=0.9690]


.. parsed-literal::

    Epoch 20/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.69it/s, loss=7.6200, acc=0.7800, kr=0.9830, val_loss=7.7290, val_acc=0.7732, val_kr=0.9858]


.. parsed-literal::

    Epoch 21/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.66it/s, loss=7.2097, acc=0.7821, kr=1.0012, val_loss=7.6041, val_acc=0.7725, val_kr=1.0047]


.. parsed-literal::

    Epoch 22/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.69it/s, loss=6.6541, acc=0.7838, kr=1.0179, val_loss=7.4834, val_acc=0.7796, val_kr=1.0211]


.. parsed-literal::

    Epoch 23/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.68it/s, loss=7.0025, acc=0.7880, kr=1.0355, val_loss=7.3719, val_acc=0.7774, val_kr=1.0357]


.. parsed-literal::

    Epoch 24/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.66it/s, loss=7.2880, acc=0.7879, kr=1.0507, val_loss=7.2474, val_acc=0.7831, val_kr=1.0540]


.. parsed-literal::

    Epoch 25/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.65it/s, loss=7.0548, acc=0.7930, kr=1.0699, val_loss=7.1355, val_acc=0.7838, val_kr=1.0715]


.. parsed-literal::

    Epoch 26/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.66it/s, loss=6.8047, acc=0.7942, kr=1.0860, val_loss=7.0455, val_acc=0.7881, val_kr=1.0870]


.. parsed-literal::

    Epoch 27/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.72it/s, loss=6.4147, acc=0.7972, kr=1.1011, val_loss=6.9661, val_acc=0.7878, val_kr=1.1017]


.. parsed-literal::

    Epoch 28/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=6.7332, acc=0.7971, kr=1.1179, val_loss=6.8569, val_acc=0.7912, val_kr=1.1194]


.. parsed-literal::

    Epoch 29/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=6.6728, acc=0.8003, kr=1.1330, val_loss=6.7894, val_acc=0.7895, val_kr=1.1355]


.. parsed-literal::

    Epoch 30/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=6.6812, acc=0.8021, kr=1.1495, val_loss=6.6940, val_acc=0.7961, val_kr=1.1507]


.. parsed-literal::

    Epoch 31/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.73it/s, loss=6.1188, acc=0.8029, kr=1.1658, val_loss=6.6080, val_acc=0.7962, val_kr=1.1674]


.. parsed-literal::

    Epoch 32/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.76it/s, loss=6.3918, acc=0.8052, kr=1.1805, val_loss=6.5385, val_acc=0.7986, val_kr=1.1812]


.. parsed-literal::

    Epoch 33/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=6.3506, acc=0.8058, kr=1.1948, val_loss=6.4665, val_acc=0.7977, val_kr=1.1968]


.. parsed-literal::

    Epoch 34/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=6.1084, acc=0.8081, kr=1.2113, val_loss=6.3989, val_acc=0.8010, val_kr=1.2104]


.. parsed-literal::

    Epoch 35/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.73it/s, loss=6.0416, acc=0.8107, kr=1.2255, val_loss=6.3671, val_acc=0.8025, val_kr=1.2271]


.. parsed-literal::

    Epoch 36/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=5.8046, acc=0.8110, kr=1.2405, val_loss=6.2684, val_acc=0.8027, val_kr=1.2384]


.. parsed-literal::

    Epoch 37/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=5.9592, acc=0.8115, kr=1.2543, val_loss=6.2166, val_acc=0.8062, val_kr=1.2518]


.. parsed-literal::

    Epoch 38/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.72it/s, loss=6.2462, acc=0.8128, kr=1.2670, val_loss=6.1399, val_acc=0.8065, val_kr=1.2688]


.. parsed-literal::

    Epoch 39/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.70it/s, loss=6.0790, acc=0.8143, kr=1.2809, val_loss=6.0902, val_acc=0.8076, val_kr=1.2820]


.. parsed-literal::

    Epoch 40/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.71it/s, loss=5.6846, acc=0.8163, kr=1.2952, val_loss=6.0258, val_acc=0.8107, val_kr=1.2934]


.. parsed-literal::

    Epoch 41/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.71it/s, loss=5.4129, acc=0.8174, kr=1.3079, val_loss=6.0086, val_acc=0.8094, val_kr=1.3056]


.. parsed-literal::

    Epoch 42/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.71it/s, loss=5.9106, acc=0.8176, kr=1.3197, val_loss=5.9359, val_acc=0.8097, val_kr=1.3216]


.. parsed-literal::

    Epoch 43/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.69it/s, loss=5.3967, acc=0.8199, kr=1.3355, val_loss=5.8801, val_acc=0.8100, val_kr=1.3338]


.. parsed-literal::

    Epoch 44/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.70it/s, loss=5.2553, acc=0.8206, kr=1.3453, val_loss=5.8445, val_acc=0.8119, val_kr=1.3451]


.. parsed-literal::

    Epoch 45/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.69it/s, loss=5.6222, acc=0.8214, kr=1.3580, val_loss=5.8061, val_acc=0.8108, val_kr=1.3577]


.. parsed-literal::

    Epoch 46/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.76it/s, loss=5.6705, acc=0.8215, kr=1.3696, val_loss=5.7312, val_acc=0.8158, val_kr=1.3675]


.. parsed-literal::

    Epoch 47/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=5.4717, acc=0.8220, kr=1.3814, val_loss=5.6903, val_acc=0.8158, val_kr=1.3799]


.. parsed-literal::

    Epoch 48/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.70it/s, loss=5.3415, acc=0.8245, kr=1.3942, val_loss=5.6476, val_acc=0.8177, val_kr=1.3909]


.. parsed-literal::

    Epoch 49/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.67it/s, loss=5.0636, acc=0.8258, kr=1.4043, val_loss=5.6481, val_acc=0.8190, val_kr=1.3989]


.. parsed-literal::

    Epoch 50/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.66it/s, loss=5.0173, acc=0.8263, kr=1.4139, val_loss=5.5660, val_acc=0.8177, val_kr=1.4114]


.. parsed-literal::

    Epoch 51/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.71it/s, loss=5.4555, acc=0.8262, kr=1.4262, val_loss=5.5223, val_acc=0.8192, val_kr=1.4247]


.. parsed-literal::

    Epoch 52/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.73it/s, loss=5.0637, acc=0.8281, kr=1.4367, val_loss=5.4895, val_acc=0.8198, val_kr=1.4339]


.. parsed-literal::

    Epoch 53/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=5.5016, acc=0.8290, kr=1.4471, val_loss=5.4692, val_acc=0.8182, val_kr=1.4462]


.. parsed-literal::

    Epoch 54/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.72it/s, loss=5.0414, acc=0.8292, kr=1.4579, val_loss=5.4138, val_acc=0.8223, val_kr=1.4535]


.. parsed-literal::

    Epoch 55/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.70it/s, loss=5.0434, acc=0.8295, kr=1.4680, val_loss=5.3816, val_acc=0.8236, val_kr=1.4648]


.. parsed-literal::

    Epoch 56/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.72it/s, loss=4.7568, acc=0.8307, kr=1.4770, val_loss=5.3524, val_acc=0.8257, val_kr=1.4728]


.. parsed-literal::

    Epoch 57/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.68it/s, loss=4.6369, acc=0.8329, kr=1.4879, val_loss=5.3317, val_acc=0.8210, val_kr=1.4853]


.. parsed-literal::

    Epoch 58/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.66it/s, loss=4.9217, acc=0.8321, kr=1.4985, val_loss=5.2853, val_acc=0.8238, val_kr=1.4971]


.. parsed-literal::

    Epoch 59/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.66it/s, loss=4.7243, acc=0.8338, kr=1.5077, val_loss=5.2464, val_acc=0.8249, val_kr=1.5054]


.. parsed-literal::

    Epoch 60/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.68it/s, loss=4.8585, acc=0.8333, kr=1.5188, val_loss=5.2077, val_acc=0.8270, val_kr=1.5132]


.. parsed-literal::

    Epoch 61/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.65it/s, loss=4.8504, acc=0.8350, kr=1.5279, val_loss=5.2295, val_acc=0.8224, val_kr=1.5253]


.. parsed-literal::

    Epoch 62/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.65it/s, loss=4.8763, acc=0.8345, kr=1.5351, val_loss=5.2056, val_acc=0.8240, val_kr=1.5308]


.. parsed-literal::

    Epoch 63/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.68it/s, loss=4.7773, acc=0.8353, kr=1.5450, val_loss=5.1410, val_acc=0.8296, val_kr=1.5429]


.. parsed-literal::

    Epoch 64/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.68it/s, loss=4.8620, acc=0.8366, kr=1.5570, val_loss=5.0879, val_acc=0.8297, val_kr=1.5494]


.. parsed-literal::

    Epoch 65/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.69it/s, loss=4.7218, acc=0.8373, kr=1.5654, val_loss=5.0698, val_acc=0.8267, val_kr=1.5599]


.. parsed-literal::

    Epoch 66/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.68it/s, loss=5.0687, acc=0.8370, kr=1.5711, val_loss=5.0683, val_acc=0.8262, val_kr=1.5710]


.. parsed-literal::

    Epoch 67/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.67it/s, loss=4.6142, acc=0.8393, kr=1.5819, val_loss=5.0545, val_acc=0.8323, val_kr=1.5746]


.. parsed-literal::

    Epoch 68/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.67it/s, loss=4.6043, acc=0.8401, kr=1.5894, val_loss=5.0035, val_acc=0.8280, val_kr=1.5857]


.. parsed-literal::

    Epoch 69/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.65it/s, loss=4.6178, acc=0.8388, kr=1.5955, val_loss=4.9740, val_acc=0.8303, val_kr=1.5918]


.. parsed-literal::

    Epoch 70/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.68it/s, loss=4.6509, acc=0.8411, kr=1.6050, val_loss=4.9418, val_acc=0.8328, val_kr=1.5971]


.. parsed-literal::

    Epoch 71/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.67it/s, loss=4.9657, acc=0.8409, kr=1.6137, val_loss=4.9159, val_acc=0.8301, val_kr=1.6092]


.. parsed-literal::

    Epoch 72/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.65it/s, loss=4.5740, acc=0.8413, kr=1.6197, val_loss=4.8943, val_acc=0.8292, val_kr=1.6132]


.. parsed-literal::

    Epoch 73/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.67it/s, loss=4.7415, acc=0.8420, kr=1.6264, val_loss=4.8727, val_acc=0.8359, val_kr=1.6174]


.. parsed-literal::

    Epoch 74/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.71it/s, loss=4.6112, acc=0.8411, kr=1.6323, val_loss=4.8517, val_acc=0.8333, val_kr=1.6285]


.. parsed-literal::

    Epoch 75/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=4.9014, acc=0.8425, kr=1.6399, val_loss=4.8309, val_acc=0.8364, val_kr=1.6357]


.. parsed-literal::

    Epoch 76/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=4.7204, acc=0.8436, kr=1.6484, val_loss=4.8106, val_acc=0.8376, val_kr=1.6430]


.. parsed-literal::

    Epoch 77/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=4.4593, acc=0.8437, kr=1.6533, val_loss=4.8434, val_acc=0.8297, val_kr=1.6484]


.. parsed-literal::

    Epoch 78/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=4.3488, acc=0.8434, kr=1.6608, val_loss=4.7688, val_acc=0.8379, val_kr=1.6522]


.. parsed-literal::

    Epoch 79/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.76it/s, loss=4.4854, acc=0.8439, kr=1.6667, val_loss=4.7244, val_acc=0.8371, val_kr=1.6608]


.. parsed-literal::

    Epoch 80/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=4.6372, acc=0.8453, kr=1.6738, val_loss=4.7326, val_acc=0.8376, val_kr=1.6623]


.. parsed-literal::

    Epoch 81/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=3.9168, acc=0.8461, kr=1.6762, val_loss=4.7040, val_acc=0.8359, val_kr=1.6727]


.. parsed-literal::

    Epoch 82/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=4.4819, acc=0.8459, kr=1.6831, val_loss=4.6923, val_acc=0.8382, val_kr=1.6779]


.. parsed-literal::

    Epoch 83/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=4.2101, acc=0.8466, kr=1.6910, val_loss=4.6570, val_acc=0.8403, val_kr=1.6814]


.. parsed-literal::

    Epoch 84/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=4.1839, acc=0.8476, kr=1.6946, val_loss=4.6394, val_acc=0.8405, val_kr=1.6863]


.. parsed-literal::

    Epoch 85/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.66it/s, loss=4.1559, acc=0.8466, kr=1.7025, val_loss=4.6774, val_acc=0.8337, val_kr=1.6948]


.. parsed-literal::

    Epoch 86/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.66it/s, loss=4.0721, acc=0.8475, kr=1.7056, val_loss=4.6079, val_acc=0.8415, val_kr=1.6987]


.. parsed-literal::

    Epoch 87/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.67it/s, loss=4.0041, acc=0.8477, kr=1.7115, val_loss=4.5860, val_acc=0.8403, val_kr=1.7020]


.. parsed-literal::

    Epoch 88/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.66it/s, loss=4.1027, acc=0.8485, kr=1.7177, val_loss=4.5727, val_acc=0.8419, val_kr=1.7091]


.. parsed-literal::

    Epoch 89/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=4.1122, acc=0.8494, kr=1.7223, val_loss=4.5689, val_acc=0.8386, val_kr=1.7194]


.. parsed-literal::

    Epoch 90/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=3.9592, acc=0.8492, kr=1.7292, val_loss=4.5552, val_acc=0.8439, val_kr=1.7182]


.. parsed-literal::

    Epoch 91/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=4.0801, acc=0.8493, kr=1.7327, val_loss=4.5449, val_acc=0.8395, val_kr=1.7275]


.. parsed-literal::

    Epoch 92/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.77it/s, loss=4.0227, acc=0.8500, kr=1.7365, val_loss=4.5189, val_acc=0.8425, val_kr=1.7322]


.. parsed-literal::

    Epoch 93/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=4.2882, acc=0.8506, kr=1.7419, val_loss=4.5133, val_acc=0.8426, val_kr=1.7356]


.. parsed-literal::

    Epoch 94/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=4.0366, acc=0.8506, kr=1.7490, val_loss=4.4820, val_acc=0.8406, val_kr=1.7410]


.. parsed-literal::

    Epoch 95/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=4.1335, acc=0.8510, kr=1.7512, val_loss=4.4676, val_acc=0.8450, val_kr=1.7446]


.. parsed-literal::

    Epoch 96/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.74it/s, loss=4.0049, acc=0.8506, kr=1.7556, val_loss=4.4764, val_acc=0.8431, val_kr=1.7505]


.. parsed-literal::

    Epoch 97/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.75it/s, loss=4.0175, acc=0.8513, kr=1.7595, val_loss=4.4278, val_acc=0.8444, val_kr=1.7504]


.. parsed-literal::

    Epoch 98/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.71it/s, loss=4.1952, acc=0.8517, kr=1.7631, val_loss=4.4052, val_acc=0.8455, val_kr=1.7555]


.. parsed-literal::

    Epoch 99/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.70it/s, loss=4.1029, acc=0.8519, kr=1.7690, val_loss=4.4153, val_acc=0.8463, val_kr=1.7614]


.. parsed-literal::

    Epoch 100/100


.. parsed-literal::

    100%|████████████████| 15/15 [00:05<00:00,  2.70it/s, loss=3.9639, acc=0.8526, kr=1.7726, val_loss=4.3812, val_acc=0.8459, val_kr=1.7586]


4. Model export
---------------

Once training is finished, the model can be optimized for inference by
using the ``vanilla_export()`` method. The ``torchlip`` layers are
converted to their PyTorch counterparts, e.g. \ ``SpectralConv2d``
layers will be converted into ``torch.nn.Conv2d`` layers.

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
      (9): Linear(in_features=64, out_features=10, bias=False)
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
    sub_data, sub_targets = iter(test_loader).next()
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

.. math::  \mathcal{M} = \frac{\text{top}_1 - \text{top}_2}{\sqrt{2}}

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
    certificates = (values[:, 0] - values[:, 1]) / np.sqrt(2)

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
    Image 0        0.541                1.74
    Image 1        1.504                3.74
    Image 2        0.406                1.35
    Image 3        0.814                1.87
    Image 4        0.154                0.54
    Image 5        0.270                0.72
    Image 6        0.191                0.72
    Image 7        0.464                1.08
    Image 8        0.906                2.31
    Image 9        0.268                0.75


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




.. image:: wasserstein_classification_fashionMNIST_files/wasserstein_classification_fashionMNIST_14_0.png
