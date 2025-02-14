import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
import lightning as pl
import timm
import pandas as pd  # Import pandas for DataFrame creation

class Regressor(pl.LightningModule):
    """
    Example of a regression model using PyTorch Lightning.

    Args:
        TIMM_MODEL (str): Pre-trained model architecture (default: "convnext_base.fb_in22k_ft_in1k").
        LEARNING_RATE (float): Learning rate for the optimizer.
        BATCH_SIZE (int): Batch size for training and validation.
        use_ema (bool): Whether to use exponential moving average (EMA) during training.
    """
    def __init__(self, TIMM_MODEL='convnext_base.fb_in22k_ft_in1k', LEARNING_RATE=1e-5, BATCH_SIZE=32, use_ema=False, criterion=nn.MSELoss):
        super().__init__()

        self.use_ema = use_ema
        self.TIMM_MODEL = TIMM_MODEL
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.model = timm.create_model(self.TIMM_MODEL, pretrained=True, in_chans=1, num_classes=1)
        self.ema = AveragedModel(self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
        self.criterion = criterion()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input data (image).

        Returns:
            torch.Tensor: Embedding (output) from the model.
        """
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer with model parameters.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
        Defines the training logic for one batch.

        Args:
            train_batch (dict): Training batch containing 'img' and 'label'.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        x, y = train_batch['img'], train_batch['cls']
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False, batch_size=self.BATCH_SIZE)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Defines the validation logic for one batch.

        Args:
            val_batch (dict): Validation batch containing 'img' and 'label'.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        x, y = val_batch['img'], val_batch['cls']
        if self.use_ema:
            logits = self.ema(x)
        else:
            logits = self(x)
        loss = self.criterion(logits, y)
        self.log('valid_loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False, batch_size=self.BATCH_SIZE)
        return loss

    def predict_step(self, val_batch, batch_idx):
        """
        Computes predictions for validation data.

        Args:
            val_batch (dict): Validation batch containing 'img' and 'label'.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        x, p = val_batch['img'], val_batch['paths']
        if self.use_ema:
            logits = self.ema(x)
        else:
            logits = self(x)

        df_logit = pd.DataFrame(logits.squeeze().cpu().numpy(), index=p, columns=[f'logit_{i}' for i in range(logits.squeeze().shape[-1])]) 
        return df_logit

    def optimizer_step(self, *args, **kwargs):
        """
        Updates optimizer parameters.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update_parameters(self.model)
