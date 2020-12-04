# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from ray.util.sgd.torch import TorchTrainer, TrainingOperator

# from torch.optim.lr_scheduler import StepLR
from ray.util.sgd.utils import BATCH_SIZE, override

from deel.torchlip.layers import (
    SpectralConv2d,
    SpectralLinear,
)

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from tqdm import trange

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


class ConvNet(nn.Module):
    def __init__(self, activation=F.relu):
        super(ConvNet, self).__init__()
        self.conv1 = SpectralConv2d(1, 32, 3, 1)
        self.conv2 = SpectralConv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = SpectralLinear(9216, 128)
        self.fc2 = SpectralLinear(128, 10)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MnistTrainingOperator(TrainingOperator):
    @override(TrainingOperator)
    def setup(self, config):
        # Create model.
        model = ConvNet()

        # Create optimizer.
        optimizer = optim.Adadelta(model.parameters(), lr=config.get("lr", 1e-2))

        # Load in training and validation data.
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # We add FileLock here because multiple workers will want to
        # download data, and this may cause overwrites since
        # DataLoader is not threadsafe.
        with FileLock(os.path.expanduser("~/data.lock")):
            train_dataset = datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            )
            test_dataset = datasets.MNIST(
                "~/data", train=False, transform=mnist_transforms
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )

        # Create scheduler.
        # scheduler = StepLR(optimizer, step_size=1, gamma=config.get("gamma"))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 250, 350], gamma=0.1
        )

        # Create loss.
        criterion = nn.CrossEntropyLoss()

        # Register all components.
        self.model, self.optimizer, self.criterion, self.scheduler = self.register(
            models=model,
            optimizers=optimizer,
            criterion=criterion,
            schedulers=scheduler,
        )
        self.register_data(train_loader=train_loader, validation_loader=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lipschitz MNIST Example")
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for connecting to the Ray cluster",
    )

    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Sets number of workers for training.",
    )

    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of epochs to train."
    )

    parser.add_argument(
        "--use-gpu", action="store_true", default=True, help="Enables GPU training"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enables FP16 training with apex. Requires `use-gpu`.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.",
    )

    parser.add_argument(
        "--tune", action="store_true", default=False, help="Tune training"
    )

    args = parser.parse_args()

    if ray.is_initialized() == False:
        ray.init(address="auto")

    trainer = TorchTrainer(
        training_operator_cls=MnistTrainingOperator,
        num_workers=args.num_workers,
        config={
            "lr": 0.1,
            "test_mode": args.smoke_test,  # subset the data
            # this will be split across workers.
            BATCH_SIZE: 64 * args.num_workers,
        },
        use_gpu=args.use_gpu,
        scheduler_step_freq="epoch",
        use_fp16=args.fp16,
        use_tqdm=False,
    )

    pbar = trange(args.num_epochs, unit="epoch")
    for i in pbar:
        info = {"num_steps": 1} if args.smoke_test else {}
        info["epoch_idx"] = i
        info["num_epochs"] = args.num_epochs
        # Increase `max_retries` to turn on fault tolerance.
        trainer.train(max_retries=1, info=info)
        val_stats = trainer.validate()
        pbar.set_postfix(dict(acc=val_stats["val_accuracy"]))

    print(trainer.validate())
    model = trainer.get_model()
    torch.save(model.state_dict(), "mnist_cnn.pth")
    trainer.shutdown()
    print("success!")
