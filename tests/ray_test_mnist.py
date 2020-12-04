# From https://github.com/pytorch/examples/blob/master/mnist/main.py with very little
#  change to add Lipschitz constraint

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from ray.tune.logger import pretty_print

from ray.util.sgd import TorchTrainer

# from ray.experimental.sgd.pytorch.pytorch_trainer import PyTorchTrainer
# ray command :./launch_ray_job.sh -c ../config/gcp_lip_pytorch.yaml -e stop
#              -j /home/justin.plakoo/deel-p-pytorch/tests/ray_test_mnist.py

from deel.lip.pt.activations import GroupSort2
from deel.lip.pt.layers import (
    SpectralConv2d,
    SpectralLinear,
)

# from deel.lip.pt.utils import evaluate_lip_const


class dummy_activation(nn.Module):
    def forward(self, X):
        return X.abs()


class Net(nn.Module):
    def __init__(self, activation=F.relu):
        super(Net, self).__init__()
        self.conv1 = SpectralConv2d(1, 32, 3, 1)
        self.conv2 = SpectralConv2d(32, 64, 3, 1)
        self.groupSort2 = GroupSort2()
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = SpectralLinear(9216, 128)
        self.fc2 = SpectralLinear(128, 10)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        x = self.groupSort2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.groupSort2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def model_creator(config):
    return Net(activation=dummy_activation())


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return optim.Adadelta(model.parameters(), lr=config.get("lr", 1e-2))


def scheduler_creator(optimizer, config):
    return StepLR(optimizer, step_size=1, gamma=config.get("gamma"))


def data_creator(config):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=config.get("batch_size", 32)
    )
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=config.get("batch_size", 32)
    )
    return train_loader, test_loader


def train(args):
    print("Train starting !!!!!!!!!!!!! {}".format(args))
    trainer1 = TorchTrainer(
        model_creator=model_creator,
        data_creator=data_creator,
        optimizer_creator=optimizer_creator,
        loss_creator=nn.NLLLoss,
        scheduler_creator=scheduler_creator,
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        config={
            "lr": 1e-2,  # used in optimizer_creator
            "hidden_size": 1,  # used in model_creator
            "gamma": args.gamma,
            "batch_size": args.batch_size,
        },
        backend="gloo",
        scheduler_step_freq="epoch",
    )
    for i in range(1, args.epochs + 1):
        print("epochs  ", i)
        print(pretty_print(trainer1.train(profile=True)))
        # print(stats)

    print(trainer1.validate())
    m = trainer1.get_model()
    print(
        "Lipschitz trained weight: % .2f, bias: % .2f"
        % (m.weight.item(), m.bias.item())
    )
    print(
        "Standard trained weight: % .2f, bias: % .2f"
        % (m.weight_origin.item(), m.bias.item())
    )
    trainer1.shutdown()
    print("success!")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=True, help="Enables GPU training"
    )
    parser.add_argument(
        "--tune", action="store_true", default=False, help="Tune training"
    )

    parser.add_argument("--reload", type=str, default=None, help="Model to reload")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.num_workers = 2
    args.pin_memory = True
    args.shuffle = True
    import ray

    ray.init()
    train(args)


if __name__ == "__main__":
    main()
