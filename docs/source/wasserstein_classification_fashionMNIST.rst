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

    epochs = 100
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
    hkr_loss = torchlip.HKRMulticlassLoss(alpha=100, min_margin=0.25)

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
            m_kr += torchlip.functional.kr_multiclass_loss(output, target)
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
        val_kr = torchlip.functional.kr_multiclass_loss(test_output, test_targets)
        val_acc = (test_output.argmax(dim=1) == test_targets.argmax(dim=1)).float().mean()

        # Validation metrics for the current epoch
        metrics += [
            f"val_{k}: {v:.04f}"
            for k, v in {
                "loss": hkr_loss(test_output, test_targets),
                "acc": (test_output.argmax(dim=1) == test_targets.argmax(dim=1))
                .float()
                .mean(),
                "KR": torchlip.functional.kr_multiclass_loss(test_output, test_targets),
            }.items()
        ]

        print(f"Epoch {epoch + 1}/{epochs}")
        print(" - ".join(metrics))



.. parsed-literal::

    Epoch 1/100
    loss: 29.2594 - acc: 0.2876 - KR: 0.0977 - val_loss: 28.3776 - val_acc: 0.5734 - val_KR: 0.1971
    Epoch 2/100
    loss: 19.7007 - acc: 0.5933 - KR: 0.2755 - val_loss: 19.3812 - val_acc: 0.5651 - val_KR: 0.3583
    Epoch 3/100
    loss: 16.0768 - acc: 0.5800 - KR: 0.4217 - val_loss: 15.7545 - val_acc: 0.5926 - val_KR: 0.4798
    Epoch 4/100
    loss: 13.6245 - acc: 0.6386 - KR: 0.5193 - val_loss: 13.5690 - val_acc: 0.6493 - val_KR: 0.5495
    Epoch 5/100
    loss: 11.9117 - acc: 0.6773 - KR: 0.5819 - val_loss: 12.2535 - val_acc: 0.6777 - val_KR: 0.6116
    Epoch 6/100
    loss: 11.2530 - acc: 0.7075 - KR: 0.6386 - val_loss: 11.4335 - val_acc: 0.6963 - val_KR: 0.6583
    Epoch 7/100
    loss: 10.7441 - acc: 0.7217 - KR: 0.6823 - val_loss: 10.8470 - val_acc: 0.7284 - val_KR: 0.6982
    Epoch 8/100
    loss: 10.3286 - acc: 0.7377 - KR: 0.7179 - val_loss: 10.3688 - val_acc: 0.7279 - val_KR: 0.7286
    Epoch 9/100
    loss: 10.0334 - acc: 0.7451 - KR: 0.7480 - val_loss: 9.9646 - val_acc: 0.7418 - val_KR: 0.7592
    Epoch 10/100
    loss: 9.2900 - acc: 0.7525 - KR: 0.7766 - val_loss: 9.6336 - val_acc: 0.7403 - val_KR: 0.7839
    Epoch 11/100
    loss: 8.8364 - acc: 0.7593 - KR: 0.8015 - val_loss: 9.3219 - val_acc: 0.7503 - val_KR: 0.8089
    Epoch 12/100
    loss: 9.1164 - acc: 0.7625 - KR: 0.8254 - val_loss: 9.0539 - val_acc: 0.7517 - val_KR: 0.8315
    Epoch 13/100
    loss: 8.1088 - acc: 0.7679 - KR: 0.8481 - val_loss: 8.8078 - val_acc: 0.7549 - val_KR: 0.8534
    Epoch 14/100
    loss: 8.4167 - acc: 0.7739 - KR: 0.8706 - val_loss: 8.5958 - val_acc: 0.7668 - val_KR: 0.8730
    Epoch 15/100
    loss: 8.2691 - acc: 0.7773 - KR: 0.8913 - val_loss: 8.3878 - val_acc: 0.7747 - val_KR: 0.8947
    Epoch 16/100
    loss: 8.0049 - acc: 0.7813 - KR: 0.9104 - val_loss: 8.1838 - val_acc: 0.7777 - val_KR: 0.9123
    Epoch 17/100
    loss: 7.6986 - acc: 0.7854 - KR: 0.9289 - val_loss: 7.9997 - val_acc: 0.7793 - val_KR: 0.9309
    Epoch 18/100
    loss: 7.3085 - acc: 0.7889 - KR: 0.9487 - val_loss: 7.8312 - val_acc: 0.7817 - val_KR: 0.9515
    Epoch 19/100
    loss: 7.2437 - acc: 0.7909 - KR: 0.9663 - val_loss: 7.6817 - val_acc: 0.7858 - val_KR: 0.9666
    Epoch 20/100
    loss: 7.1865 - acc: 0.7936 - KR: 0.9857 - val_loss: 7.5286 - val_acc: 0.7862 - val_KR: 0.9879
    Epoch 21/100
    loss: 7.4875 - acc: 0.7951 - KR: 1.0032 - val_loss: 7.3780 - val_acc: 0.7882 - val_KR: 1.0076
    Epoch 22/100
    loss: 6.8301 - acc: 0.7982 - KR: 1.0202 - val_loss: 7.2544 - val_acc: 0.7917 - val_KR: 1.0218
    Epoch 23/100
    loss: 6.8855 - acc: 0.8000 - KR: 1.0381 - val_loss: 7.1429 - val_acc: 0.7927 - val_KR: 1.0406
    Epoch 24/100
    loss: 6.8564 - acc: 0.8029 - KR: 1.0552 - val_loss: 7.0058 - val_acc: 0.7942 - val_KR: 1.0583
    Epoch 25/100
    loss: 6.5540 - acc: 0.8049 - KR: 1.0736 - val_loss: 6.9071 - val_acc: 0.7954 - val_KR: 1.0744
    Epoch 26/100
    loss: 6.9353 - acc: 0.8060 - KR: 1.0891 - val_loss: 6.8090 - val_acc: 0.7973 - val_KR: 1.0902
    Epoch 27/100
    loss: 6.2051 - acc: 0.8078 - KR: 1.1073 - val_loss: 6.7187 - val_acc: 0.7993 - val_KR: 1.1065
    Epoch 28/100
    loss: 6.2606 - acc: 0.8090 - KR: 1.1237 - val_loss: 6.6394 - val_acc: 0.8014 - val_KR: 1.1215
    Epoch 29/100
    loss: 6.8432 - acc: 0.8113 - KR: 1.1393 - val_loss: 6.5825 - val_acc: 0.7986 - val_KR: 1.1387
    Epoch 30/100
    loss: 6.3484 - acc: 0.8133 - KR: 1.1548 - val_loss: 6.4634 - val_acc: 0.8052 - val_KR: 1.1519
    Epoch 31/100
    loss: 5.8132 - acc: 0.8145 - KR: 1.1706 - val_loss: 6.3965 - val_acc: 0.8053 - val_KR: 1.1698
    Epoch 32/100
    loss: 5.9282 - acc: 0.8154 - KR: 1.1860 - val_loss: 6.3272 - val_acc: 0.8073 - val_KR: 1.1864
    Epoch 33/100
    loss: 6.2292 - acc: 0.8169 - KR: 1.2001 - val_loss: 6.2783 - val_acc: 0.8082 - val_KR: 1.1993
    Epoch 34/100
    loss: 5.8215 - acc: 0.8182 - KR: 1.2161 - val_loss: 6.2135 - val_acc: 0.8109 - val_KR: 1.2126
    Epoch 35/100
    loss: 5.8808 - acc: 0.8181 - KR: 1.2292 - val_loss: 6.1369 - val_acc: 0.8113 - val_KR: 1.2249
    Epoch 36/100
    loss: 5.8833 - acc: 0.8205 - KR: 1.2430 - val_loss: 6.0850 - val_acc: 0.8119 - val_KR: 1.2448
    Epoch 37/100
    loss: 5.6469 - acc: 0.8224 - KR: 1.2582 - val_loss: 6.0367 - val_acc: 0.8127 - val_KR: 1.2576
    Epoch 38/100
    loss: 5.3902 - acc: 0.8232 - KR: 1.2731 - val_loss: 5.9918 - val_acc: 0.8122 - val_KR: 1.2707
    Epoch 39/100
    loss: 5.5306 - acc: 0.8233 - KR: 1.2877 - val_loss: 5.9173 - val_acc: 0.8157 - val_KR: 1.2828
    Epoch 40/100
    loss: 5.7492 - acc: 0.8236 - KR: 1.3006 - val_loss: 5.8784 - val_acc: 0.8162 - val_KR: 1.2966
    Epoch 41/100
    loss: 5.6263 - acc: 0.8254 - KR: 1.3124 - val_loss: 5.8362 - val_acc: 0.8139 - val_KR: 1.3108
    Epoch 42/100
    loss: 5.0626 - acc: 0.8259 - KR: 1.3260 - val_loss: 5.7580 - val_acc: 0.8187 - val_KR: 1.3245
    Epoch 43/100
    loss: 5.4969 - acc: 0.8266 - KR: 1.3394 - val_loss: 5.7457 - val_acc: 0.8140 - val_KR: 1.3360
    Epoch 44/100
    loss: 5.3117 - acc: 0.8277 - KR: 1.3487 - val_loss: 5.6837 - val_acc: 0.8174 - val_KR: 1.3481
    Epoch 45/100
    loss: 5.7271 - acc: 0.8282 - KR: 1.3632 - val_loss: 5.6208 - val_acc: 0.8203 - val_KR: 1.3608
    Epoch 46/100
    loss: 5.1668 - acc: 0.8292 - KR: 1.3752 - val_loss: 5.5900 - val_acc: 0.8213 - val_KR: 1.3708
    Epoch 47/100
    loss: 4.9962 - acc: 0.8296 - KR: 1.3862 - val_loss: 5.5538 - val_acc: 0.8227 - val_KR: 1.3815
    Epoch 48/100
    loss: 5.5416 - acc: 0.8302 - KR: 1.3957 - val_loss: 5.5073 - val_acc: 0.8210 - val_KR: 1.3916
    Epoch 49/100
    loss: 5.1352 - acc: 0.8315 - KR: 1.4077 - val_loss: 5.4572 - val_acc: 0.8233 - val_KR: 1.4030
    Epoch 50/100
    loss: 5.2471 - acc: 0.8304 - KR: 1.4180 - val_loss: 5.4316 - val_acc: 0.8212 - val_KR: 1.4161
    Epoch 51/100
    loss: 5.2000 - acc: 0.8318 - KR: 1.4331 - val_loss: 5.4180 - val_acc: 0.8260 - val_KR: 1.4260
    Epoch 52/100
    loss: 4.9510 - acc: 0.8332 - KR: 1.4394 - val_loss: 5.3918 - val_acc: 0.8212 - val_KR: 1.4347
    Epoch 53/100
    loss: 4.9898 - acc: 0.8324 - KR: 1.4491 - val_loss: 5.3414 - val_acc: 0.8249 - val_KR: 1.4484
    Epoch 54/100
    loss: 4.9740 - acc: 0.8337 - KR: 1.4627 - val_loss: 5.2889 - val_acc: 0.8285 - val_KR: 1.4580
    Epoch 55/100
    loss: 5.0518 - acc: 0.8352 - KR: 1.4750 - val_loss: 5.2474 - val_acc: 0.8268 - val_KR: 1.4684
    Epoch 56/100
    loss: 4.7321 - acc: 0.8362 - KR: 1.4824 - val_loss: 5.2632 - val_acc: 0.8311 - val_KR: 1.4791
    Epoch 57/100
    loss: 4.6002 - acc: 0.8372 - KR: 1.4899 - val_loss: 5.1873 - val_acc: 0.8277 - val_KR: 1.4849
    Epoch 58/100
    loss: 4.9440 - acc: 0.8369 - KR: 1.5031 - val_loss: 5.2002 - val_acc: 0.8308 - val_KR: 1.4942
    Epoch 59/100
    loss: 4.7580 - acc: 0.8380 - KR: 1.5101 - val_loss: 5.1254 - val_acc: 0.8293 - val_KR: 1.5015
    Epoch 60/100
    loss: 4.5367 - acc: 0.8382 - KR: 1.5165 - val_loss: 5.1236 - val_acc: 0.8303 - val_KR: 1.5167
    Epoch 61/100
    loss: 4.9528 - acc: 0.8394 - KR: 1.5295 - val_loss: 5.0781 - val_acc: 0.8311 - val_KR: 1.5254
    Epoch 62/100
    loss: 4.7571 - acc: 0.8376 - KR: 1.5367 - val_loss: 5.0972 - val_acc: 0.8263 - val_KR: 1.5303
    Epoch 63/100
    loss: 4.9513 - acc: 0.8381 - KR: 1.5449 - val_loss: 5.0313 - val_acc: 0.8310 - val_KR: 1.5422
    Epoch 64/100
    loss: 4.7990 - acc: 0.8408 - KR: 1.5556 - val_loss: 5.0695 - val_acc: 0.8326 - val_KR: 1.5489
    Epoch 65/100
    loss: 4.4465 - acc: 0.8415 - KR: 1.5639 - val_loss: 4.9820 - val_acc: 0.8339 - val_KR: 1.5594
    Epoch 66/100
    loss: 4.6970 - acc: 0.8414 - KR: 1.5710 - val_loss: 4.9286 - val_acc: 0.8342 - val_KR: 1.5655
    Epoch 67/100
    loss: 4.8138 - acc: 0.8424 - KR: 1.5783 - val_loss: 4.9270 - val_acc: 0.8363 - val_KR: 1.5722
    Epoch 68/100
    loss: 4.3654 - acc: 0.8421 - KR: 1.5838 - val_loss: 4.9229 - val_acc: 0.8352 - val_KR: 1.5788
    Epoch 69/100
    loss: 4.4744 - acc: 0.8424 - KR: 1.5959 - val_loss: 4.8905 - val_acc: 0.8318 - val_KR: 1.5876
    Epoch 70/100
    loss: 4.1829 - acc: 0.8419 - KR: 1.6027 - val_loss: 4.8706 - val_acc: 0.8329 - val_KR: 1.5967
    Epoch 71/100
    loss: 4.5686 - acc: 0.8435 - KR: 1.6103 - val_loss: 4.8166 - val_acc: 0.8390 - val_KR: 1.6024
    Epoch 72/100
    loss: 4.5110 - acc: 0.8450 - KR: 1.6153 - val_loss: 4.7979 - val_acc: 0.8378 - val_KR: 1.6082
    Epoch 73/100
    loss: 4.2965 - acc: 0.8450 - KR: 1.6246 - val_loss: 4.7684 - val_acc: 0.8371 - val_KR: 1.6181
    Epoch 74/100
    loss: 4.5045 - acc: 0.8458 - KR: 1.6317 - val_loss: 4.7529 - val_acc: 0.8370 - val_KR: 1.6209
    Epoch 75/100
    loss: 4.2155 - acc: 0.8464 - KR: 1.6355 - val_loss: 4.7756 - val_acc: 0.8345 - val_KR: 1.6299
    Epoch 76/100
    loss: 4.5120 - acc: 0.8456 - KR: 1.6458 - val_loss: 4.7113 - val_acc: 0.8375 - val_KR: 1.6363
    Epoch 77/100
    loss: 4.0661 - acc: 0.8473 - KR: 1.6511 - val_loss: 4.7197 - val_acc: 0.8382 - val_KR: 1.6477
    Epoch 78/100
    loss: 4.6509 - acc: 0.8472 - KR: 1.6572 - val_loss: 4.6846 - val_acc: 0.8405 - val_KR: 1.6523
    Epoch 79/100
    loss: 4.4285 - acc: 0.8473 - KR: 1.6653 - val_loss: 4.6586 - val_acc: 0.8393 - val_KR: 1.6576
    Epoch 80/100
    loss: 4.1109 - acc: 0.8490 - KR: 1.6714 - val_loss: 4.6212 - val_acc: 0.8429 - val_KR: 1.6618
    Epoch 81/100
    loss: 4.3080 - acc: 0.8479 - KR: 1.6744 - val_loss: 4.6023 - val_acc: 0.8423 - val_KR: 1.6677
    Epoch 82/100
    loss: 4.4364 - acc: 0.8488 - KR: 1.6819 - val_loss: 4.5975 - val_acc: 0.8429 - val_KR: 1.6796
    Epoch 83/100
    loss: 4.2708 - acc: 0.8496 - KR: 1.6891 - val_loss: 4.5956 - val_acc: 0.8397 - val_KR: 1.6841
    Epoch 84/100
    loss: 4.0521 - acc: 0.8487 - KR: 1.6957 - val_loss: 4.5549 - val_acc: 0.8419 - val_KR: 1.6866
    Epoch 85/100
    loss: 4.1555 - acc: 0.8496 - KR: 1.6989 - val_loss: 4.5356 - val_acc: 0.8441 - val_KR: 1.6897
    Epoch 86/100
    loss: 4.4707 - acc: 0.8508 - KR: 1.7069 - val_loss: 4.5050 - val_acc: 0.8437 - val_KR: 1.6984
    Epoch 87/100
    loss: 4.2618 - acc: 0.8509 - KR: 1.7085 - val_loss: 4.5196 - val_acc: 0.8413 - val_KR: 1.7047
    Epoch 88/100
    loss: 4.2785 - acc: 0.8503 - KR: 1.7216 - val_loss: 4.4813 - val_acc: 0.8412 - val_KR: 1.7114
    Epoch 89/100
    loss: 4.5094 - acc: 0.8510 - KR: 1.7232 - val_loss: 4.4764 - val_acc: 0.8447 - val_KR: 1.7143
    Epoch 90/100
    loss: 4.5493 - acc: 0.8521 - KR: 1.7263 - val_loss: 4.4730 - val_acc: 0.8456 - val_KR: 1.7227
    Epoch 91/100
    loss: 4.0957 - acc: 0.8520 - KR: 1.7338 - val_loss: 4.4374 - val_acc: 0.8469 - val_KR: 1.7242
    Epoch 92/100
    loss: 3.6583 - acc: 0.8528 - KR: 1.7394 - val_loss: 4.4136 - val_acc: 0.8458 - val_KR: 1.7300
    Epoch 93/100
    loss: 3.7080 - acc: 0.8522 - KR: 1.7466 - val_loss: 4.4027 - val_acc: 0.8465 - val_KR: 1.7382
    Epoch 94/100
    loss: 4.0064 - acc: 0.8534 - KR: 1.7492 - val_loss: 4.3931 - val_acc: 0.8471 - val_KR: 1.7391
    Epoch 95/100
    loss: 4.0135 - acc: 0.8534 - KR: 1.7537 - val_loss: 4.3785 - val_acc: 0.8458 - val_KR: 1.7465
    Epoch 96/100
    loss: 4.0046 - acc: 0.8536 - KR: 1.7581 - val_loss: 4.3515 - val_acc: 0.8461 - val_KR: 1.7522
    Epoch 97/100
    loss: 4.2858 - acc: 0.8546 - KR: 1.7638 - val_loss: 4.3474 - val_acc: 0.8455 - val_KR: 1.7583
    Epoch 98/100
    loss: 3.7936 - acc: 0.8546 - KR: 1.7694 - val_loss: 4.3374 - val_acc: 0.8448 - val_KR: 1.7613
    Epoch 99/100
    loss: 3.9562 - acc: 0.8550 - KR: 1.7742 - val_loss: 4.3467 - val_acc: 0.8484 - val_KR: 1.7600
    Epoch 100/100
    loss: 4.1584 - acc: 0.8557 - KR: 1.7760 - val_loss: 4.2946 - val_acc: 0.8490 - val_KR: 1.7719


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
    Image 0        0.619                1.76
    Image 1        1.454                3.57
    Image 2        0.515                1.51
    Image 3        0.891                2.02
    Image 4        0.117                0.32
    Image 5        0.260                0.64
    Image 6        0.161                0.57
    Image 7        0.519                1.17
    Image 8        0.955                2.51
    Image 9        0.264                0.70


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
