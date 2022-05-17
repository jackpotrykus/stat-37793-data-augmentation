from fastshap.image_surrogate import ImageSurrogate
from fastshap.utils import MaskLayer2d, KLDivLoss, DatasetInputOnly
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from copy import deepcopy
from resnet import ResNet18


class fastshap_wrapper:
    def __init__(
        self,
        model,
        train_set,
        val_set,
        surrogate_epochs=20,
        explainer_epochs=20,
        experiment_name="",
    ):
        self.model = model
        torch.manual_seed(0)
        # Select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # train surrogate model
        # Check for model
        if os.path.isfile(experiment_name+'cifar surrogate.pt'):
            print('Loading saved surrogate model')
            surr = torch.load(experiment_name+'cifar surrogate.pt').to(device)
            self.surrogate = ImageSurrogate(surr, width=32, height=32, superpixel_size=2)

        else:
            # Create model
            surr = nn.Sequential(
                MaskLayer2d(value=0, append=True),
                ResNet18(in_channels=4, num_classes=10),
            ).to(device)

            # Set up surrogate object
            self.surrogate = ImageSurrogate(surr, width=32, height=32, superpixel_size=2)

            # Set up datasets
            train_surr = DatasetInputOnly(train_set)
            val_surr = DatasetInputOnly(val_set)
            original_model = nn.Sequential(model, nn.Softmax(dim=1))

            # Train
            self.surrogate.train_original_model(
                train_surr,
                val_surr,
                original_model,
                batch_size=256,
                max_epochs=surrogate_epochs,
                loss_fn=KLDivLoss(),
                lookback=10,
                bar=True,
                verbose=True,
            )

            # Save surrogate
            surr.cpu()
            torch.save(surr, experiment_name + "cifar surrogate.pt")
            surr.to(device)
            

        from unet import UNet
        from fastshap import FastSHAP

        # Check for model
        if os.path.isfile(experiment_name+'cifar explainer.pt'):
            print('Loading saved explainer model')
            explainer = torch.load(experiment_name+'cifar explainer.pt').to(device)
            fastshap = FastSHAP(explainer, self.surrogate, link=nn.LogSoftmax(dim=1))
            self.fastshap = fastshap
        else:
            # Set up explainer model
            explainer = UNet(n_classes=10, num_down=2, num_up=1, num_convs=3).to(device)

            # Set up FastSHAP object
            fastshap = FastSHAP(explainer, self.surrogate, link=nn.LogSoftmax(dim=1))

            # Set up datasets
            fastshap_train = DatasetInputOnly(train_set)
            fastshap_val = DatasetInputOnly(val_set)

            # Train
            fastshap.train(
                fastshap_train,
                fastshap_val,
                batch_size=128,
                num_samples=2,
                max_epochs=explainer_epochs,
                eff_lambda=1e-2,
                validation_samples=1,
                lookback=10,
                bar=True,
                verbose=True,
            )

            # Save explainer
            explainer.cpu()
            torch.save(explainer, experiment_name + "cifar explainer.pt")
            explainer.to(device)
            self.fastshap = fastshap

    def plot_results(self, val_set, return_raw=False, num_samples=10):
        # plot results
        np.random.seed(42)
        import matplotlib.pyplot as plt
#         device = next(self.fastshap.explainer.parameters()).device
        # Select one image from each class
        dset = val_set
        targets = np.array(dset.targets)
        num_classes = targets.max() + 1
        inds_lists = [np.where(targets == cat)[0] for cat in range(num_classes)]
        if return_raw:
            inds = np.concatenate([np.random.choice(cat_inds, size=num_samples) for cat_inds in inds_lists])
        else:
            inds = [np.random.choice(cat_inds) for cat_inds in inds_lists]
        x, y = zip(*[dset[ind] for ind in inds])
        x = torch.stack(x)

        # Get explanations
        values = self.fastshap.shap_values(x.to(self.device))
        
        if return_raw:
            shapley_values = []
            for ind,a in enumerate(y):
                shapley_values.append(self.fastshap.shap_values(x)[ind,a,:,:])
            shapley_values = np.stack(shapley_values)
            return values
        
        # Get predictions
        pred = self.surrogate(
            x.to(self.device),
            torch.ones(num_classes, self.surrogate.num_players, device=self.device)
        ).softmax(dim=1).cpu().data.numpy()

        fig, axarr = plt.subplots(num_classes, num_classes + 1, figsize=(22, 20))

        for row in range(num_classes):
            # Image
            classes = [
                "Airplane",
                "Car",
                "Bird",
                "Cat",
                "Deer",
                "Dog",
                "Frog",
                "Horse",
                "Ship",
                "Truck",
            ]
            mean = np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis]
            std = np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis]
            im = x[row].numpy() * std + mean
            im = im.transpose(1, 2, 0).astype(float)
            im = np.clip(im, a_min=0, a_max=1)
            axarr[row, 0].imshow(im, vmin=0, vmax=1)
            axarr[row, 0].set_xticks([])
            axarr[row, 0].set_yticks([])
            axarr[row, 0].set_ylabel("{}".format(classes[y[row]]), fontsize=14)

            # Explanations
            m = np.abs(values[row]).max()
            for col in range(num_classes):
                axarr[row, col + 1].imshow(
                    values[row, col], cmap="seismic", vmin=-m, vmax=m
                )
                axarr[row, col + 1].set_xticks([])
                axarr[row, col + 1].set_yticks([])
                if col == y[row]:
                    axarr[row, col + 1].set_xlabel(
                        "{:.2f}".format(pred[row, col]), fontsize=12, fontweight="bold"
                    )
                else:
                    axarr[row, col + 1].set_xlabel(
                        "{:.2f}".format(pred[row, col]), fontsize=12
                    )

                # Class labels
                if row == 0:
                    axarr[row, col + 1].set_title(
                        "{}".format(classes[y[col]]), fontsize=14
                    )

        plt.tight_layout()
        plt.show()
