
import os

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR


class StoryModule(pl.LightningModule):
    """
    Attributes:
        model: BART model
        total_steps: total training steps for lr scheduling
        max_learning_rate: Max LR
        min_learning_rate: Min LR
        warmup_rate: warmup step rate
        model_save_dir: path to save model
        r3f_lambda: R3F parameter
    """

    def __init__(
        self,
        model,
        model_save_dir,
        total_steps,
        max_learning_rate: float = 2e-4,
        min_learning_rate: float = 2e-5,
        warmup_rate: float = 0.1,
        r3f_lambda: float = 0.1
    ):
        super().__init__()

        self.model = model
        self.total_steps = total_steps
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_rate = warmup_rate
        self.model_save_dir = model_save_dir
        self.r3f_lambda = r3f_lambda
        self.validation_step_loss = []

        self.save_hyperparameters(
            {
                **model.config.to_dict(),
                "total_steps": total_steps,
                "max_learning_rate": self.max_learning_rate,
                "min_learning_rate": self.min_learning_rate,
                "warmup_rate": self.warmup_rate,
                "r3f_lambda": self.r3f_lambda,
            }
        )

    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )

        labels = batch["decoder_input_ids"][:, 1:].reshape(-1)
        logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])

        loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id)
        metrics = {"loss": loss}
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )

        labels = batch["decoder_input_ids"][:, 1:].reshape(-1)
        logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])

        loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id)
        metrics = {"loss(v)": loss}
        self.validation_step_loss.append(loss)

        # metrics["accuracy(v)"] = accuracy(logits,
        #                                   labels,
        #                                   task="multiclass",
        #                                   ignore_index=self.model.config.pad_token_id)
        
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return metrics

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.max_learning_rate)
        scheduler = CyclicLR(
            optimizer,
            base_lr=self.min_learning_rate,
            max_lr=self.max_learning_rate,
            step_size_up=int(self.total_steps * self.warmup_rate),
            step_size_down=self.total_steps - int(self.total_steps * self.warmup_rate),
            mode='triangular',
            cycle_momentum=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "Learning Rate"
            },
        }

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            losses = [output.mean() for output in self.validation_step_loss]
            loss_mean = sum(losses) / len(losses)

            self.model.save_pretrained(
                os.path.join(
                    self.model_save_dir,
                    f"model-{self.current_epoch:02d}epoch-{self.global_step}steps-{loss_mean:.4f}loss",
                ),
            )

        self.validation_step_loss.clear()  # free memory
