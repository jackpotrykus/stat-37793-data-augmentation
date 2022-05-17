import torch
import torchvision
from dataclasses import dataclass, field
import tqdm
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from copy import deepcopy
from resnet import ResNet18

CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class CIFAR10AugmentationExperiment:
    """Wrapper class for conducting CIFAR10 Data Augmentation experiments

    Attributes
    ----------
    transforms: torchvision.transforms.Compose
      Transformations/Augmentations to apply to the data. The only parameter required to be passed to `__init__`.
    train_set: torchvision.datasets.cifar.CIFAR10
      Training data, downloaded upon instantiation. Transforms are applied to ONLY THIS DATA
    test: torchvision.datasets.cifar.CIFAR10
      Evaluation data, downloaded upon instantiation
    """
    torch.manual_seed(0)
    
    # User only needs to supply this
    transforms: torchvision.transforms.Compose

    # THESE ARE NOT SET BY THE USER... they are set in `__post_init__`
    train_set: torchvision.datasets.cifar.CIFAR10 = field(init=False)
    test_set: torchvision.datasets.cifar.CIFAR10 = field(init=False)
    model: torchvision.models.resnet.ResNet = field(init=False)

    def __post_init__(self) -> None:
        """Downloads data and applies transforms to TRAIN DATA ONLY. Also initializes `self.model`"""
        # fmt: off
        
        self.train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=self.transforms)
        self.test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ))
        # fmt: on

        # Initialize an untrained ResNet18 model in `self.model`
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet18(num_classes=10).to(device)

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, nepoch: int = 10, batch_size: int = 4, experiment_name: str = '', *args, **kwargs):
        """Train the ResNet18 model in `self.model`, using `self.train_set`

        nepoch: int, default=10
          Number of epochs to train for
        batch_size: int, default=4
          Batch size for the `DataLoader` when training
        """
        train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            *args,
            **kwargs,
        )
        self.train_hist = []
        self.val_hist = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        lr = 1e-3
        mbsize = batch_size  # 16
        max_nepochs = nepoch
        loss_fn = nn.CrossEntropyLoss()
        lookback = 10
        verbose = True

        # Validation function
        val_loader = DataLoader(
            self.test_set, batch_size=mbsize, shuffle=False, num_workers=4
        )

        def validate(model):
            n = 0
            mean_loss = 0
            mean_acc = 0

            for x, y in val_loader:
                # Move to GPU.
                n += len(x)
                x = x.to(device)
                y = y.to(device)

                # Get predictions.
                pred = model(x)

                # Update loss.
                loss = loss_fn(pred, y).item()
                mean_loss += len(x) * (loss - mean_loss) / n

                # Update accuracy.
                acc = (torch.argmax(pred, dim=1) == y).float().mean().item()
                mean_acc += len(x) * (acc - mean_acc) / n

            return mean_loss, mean_acc

        # Data loader
        train_loader = DataLoader(
            self.train_set,
            batch_size=mbsize,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

        # Setup
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=lookback // 2,
            min_lr=1e-5,
            mode="max",
            verbose=verbose,
        )
        loss_list = []
        acc_list = []
        min_criterion = np.inf
        min_epoch = 0

        # Train
        for epoch in range(max_nepochs):
            n = 0
            mean_loss = 0
            mean_acc = 0
            for x, y in tqdm(train_loader, desc="Training loop", leave=True):
                # Move to device.
                x = x.to(device=device)
                y = y.to(device=device)

                # Take gradient step.
                loss = loss_fn(self.model(x), y)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                n += len(x)
                x = x.to(device)
                y = y.to(device)

                # Get predictions.
                pred = self.model(x)

                # Update loss.
                loss = loss_fn(pred, y).item()
                mean_loss += len(x) * (loss - mean_loss) / n

                # Update accuracy.
                acc = (torch.argmax(pred, dim=1) == y).float().mean().item()
                mean_acc += len(x) * (acc - mean_acc) / n

            # Check progress.
            self.train_hist.append(mean_acc)
            with torch.no_grad():
                # Calculate validation loss.
                self.model.eval()
                val_loss, val_acc = validate(self.model)
                self.model.train()
                if verbose:
                    print("----- Epoch = {} -----".format(epoch + 1))
                    print("Val loss = {:.4f}".format(val_loss))
                    print("Val acc = {:.4f}".format(val_acc))
                loss_list.append(val_loss)
                acc_list.append(val_acc)
                self.val_hist.append(val_acc)
                scheduler.step(val_acc)

                # Check convergence criterion.
                val_criterion = -val_acc
                if val_criterion < min_criterion:
                    min_criterion = val_criterion
                    min_epoch = epoch
                    best_model = deepcopy(self.model)
                    print("")
                    print("New best epoch, acc = {:.4f}".format(val_acc))
                    print("")
                elif (epoch - min_epoch) == lookback:
                    if verbose:
                        print("Stopping early")
                    break

        # Keep best model
        self.model = best_model

        # Save model
        self.model.cpu()
        torch.save(self.model, experiment_name+"cifar resnet.pt")
        self.model.to(device)

    def evaluate_model(self) -> None:
        """Evaluate the model. Predict model over test set and produce diagnostic plots, tables, etc."""
        raise NotImplementedError


def example_usage() -> CIFAR10AugmentationExperiment:
    """Example usage of `CIFAR10AugmentationExperiment`"""
    # Example usage of the above
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    c = CIFAR10AugmentationExperiment(transform)
    c.train(nepoch=1, batch_size=4)
    print(type(c.model))
    # c.evaluate_model  # NOT IMPLEMENTED

    return c
