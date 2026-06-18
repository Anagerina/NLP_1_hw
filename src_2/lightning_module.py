import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from src_2.model import CNNBiLSTM


class CNNBiLSTMModule(pl.LightningModule):
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
        self.model = CNNBiLSTM(
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

    def on_train_epoch_start(self):
        if self.current_epoch == 6:
            print(
                "\n=== [Lightning] РАЗМОРОЗКА FASTTEXT ЭМБЕДДИНГОВ ДЛЯ FINE-TUNING ==="
            )

            if hasattr(self.model, "embedding"):
                self.model.embedding.weight.requires_grad = True
            elif hasattr(self, "embedding"):
                self.embedding.weight.requires_grad = True

            for param_group in self.optimizers().param_groups:
                param_group["lr"] = 5e-6

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=30,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
