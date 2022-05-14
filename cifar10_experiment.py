import torch
import torchvision
from dataclasses import dataclass, field
import tqdm
from torch.utils.data.dataloader import DataLoader


CIFAR10_CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


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
        self.test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
        # fmt: on

        # Initialize an untrained ResNet18 model in `self.model`
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)

    def train(self, nepoch: int = 10, batch_size: int = 4, *args, **kwargs):
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

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Train for `nepoch` loops... tqdm progress bar added to track
        for epoch in tqdm.trange(nepoch):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

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
