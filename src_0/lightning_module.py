import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from src_0.model import BiLSTM


class BiLSTMModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        hidden_dim=128,
        num_layers=1,
        num_classes=4,
        dropout=0.3,
        lr=1e-3,
        embedding_matrix=None,
        freeze_embeddings=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_matrix"])
        self.model = BiLSTM(
            vocab_size,
            embed_dim,
            hidden_dim,
            num_layers,
            num_classes,
            dropout,
            embedding_matrix,
            freeze_embeddings,
        )
        self.lr = lr

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        logits = self(x, mask)
        loss = self.criterion(logits, y)

        self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        logits = self(x, mask)
        loss = self.criterion(logits, y)

        self.val_acc(logits, y)
        self.val_f1(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)

        if (
            hasattr(self.trainer, "estimated_stepping_batches")
            and self.trainer.estimated_stepping_batches
        ):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = self.trainer.max_epochs * 1000

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 10,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
