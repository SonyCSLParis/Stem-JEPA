from copy import deepcopy
from typing import Any, Dict, Tuple

import wandb

import torch
from lightning import LightningModule

from src.callbacks.ma_update import MAWeightUpdate
from src.utils.entropy import compute_batch_entropy


class MixStemsModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self,
            encoder: torch.nn.Module,
            predictor: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler = None,
            ma_callback: MAWeightUpdate = MAWeightUpdate(),
            base_lr: float = 0.0003,
            compile: bool = False,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param encoder: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model using `torch.compile`
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["encoder", "predictor", "criterion", "optimizer", "scheduler", "ma_callback"])

        # context encoder
        self.encoder = encoder

        # target encoder as a moving average of the context encoder
        self.target_encoder = deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.ma_callback = ma_callback

        self.predictor = predictor

        # loss function
        self.criterion = criterion

        # optimizer and scheduler
        self.optimizer_cls = optimizer
        self.scheduler_cls = scheduler

    def on_train_batch_end(self, outputs: torch.Tensor, batch: torch.Tensor, batch_idx: int) -> None:
        self.ma_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.encoder(x)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        ctx, tgt, idx = batch
        latents = self.encoder(ctx)
        preds = self.predictor(latents, idx)

        with torch.no_grad():
            targets = self.target_encoder(tgt)

        loss = self.criterion(preds, targets)

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.predictor = torch.compile(self.predictor)
            self.target_encoder = torch.compile(self.target_encoder)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        parameters = list(self.encoder.parameters()) + list(self.predictor.parameters())
        optimizer = self.optimizer_cls(params=parameters)
        if self.scheduler_cls is not None:
            try:
                scheduler = self.scheduler_cls(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {  # TODO: adapt this to LinearWarmup thingy
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            except TypeError:
                scheduler = self.scheduler_cls(optimizer=optimizer, trainer=self.trainer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1
                    }
                }
        return {"optimizer": optimizer}
