import logging

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


def tuner(training, train_dataloader, val_dataloader, hidden_sizes, limit_train_batches):
    models = []
    for hidden_size in hidden_sizes:
        for limit_train_batch in limit_train_batches:
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=2, mode="min")
            trainer = pl.Trainer(
                max_epochs=25,
                enable_model_summary=True,
                gradient_clip_val=0.1,
                limit_train_batches=limit_train_batch,
                callbacks=[early_stop_callback],
            )
            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=0.01,
                hidden_size=hidden_size,
                attention_head_size=2,
                lstm_layers=2,
                dropout=0.1,
                hidden_continuous_size=8,
                loss=QuantileLoss(),
                optimizer="Ranger",
                reduce_on_plateau_patience=4,
            )
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            best_tft = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
            models.append({"hidden_size": hidden_size, "limit_train_batches": limit_train_batch, "model": best_tft})
    return models
