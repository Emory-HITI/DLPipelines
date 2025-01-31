import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
import lightning as pl  # Import the PyTorch Lightning library
import timm
import pandas as pd  # Import pandas for DataFrame creation
from segmentation_models_pytorch import Unet, MAnet  # Import segmentation models

class Segmentation(pl.LightningModule):
    """
    Example of a model for image segmentation using PyTorch Lightning.
    """

    def __init__(self, LEARNING_RATE=1e-5, BATCH_SIZE=32):
        super().__init__()
        # Initialize the segmentation model (you can choose between Unet or MAnet)
        self.model = MAnet(encoder_name="resnet50", in_channels=1, classes=1, encoder_weights='imagenet', activation='sigmoid')
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE

    def forward(self, x):
        """
        Forward pass through the segmentation model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Embedded features (segmentation output).
        """
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
        Training step for the model.

        Args:
            train_batch (dict): Training batch containing 'img' and 'msk' keys.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        x, y = train_batch['img'], train_batch['msk']
        logits = self(x)
        y = y.float()
        criterion = DiceLoss()  # Assuming DiceLoss is defined elsewhere
        loss = criterion(logits, y)
        self.log('train_loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False, batch_size=self.BATCH_SIZE)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Validation step for the model.

        Args:
            val_batch (dict): Validation batch containing 'img' and 'msk' keys.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        x, y = val_batch['img'], val_batch['msk']
        logits = self(x)
        y = y.float()
        criterion = DiceLoss()  # Assuming DiceLoss is defined elsewhere
        loss = criterion(logits, y)
        self.log('valid_loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False, batch_size=self.BATCH_SIZE)
        return loss
