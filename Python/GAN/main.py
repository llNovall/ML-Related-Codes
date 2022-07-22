from GANDataLoader import GANDataModule
from GANModel import GAN
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import wandb

if __name__ == '__main__':

    data_location = 'Data/'

    checkpoint_callback = ModelCheckpoint(dirpath='Checkpoints', filename='{epoch}-{train_steps}',
                                          auto_insert_metric_name=True, every_n_train_steps=500,
                                          save_last=True)

    model = GAN(d_hidden_size=64, g_latent_dim=100, g_hidden_size=64,
                learning_rate=0.002, beta=(0.5, 0.999),
                weight_decay=0.0, sample_interval=1000)

    model.initialize_weights()

    dm = GANDataModule(path=data_location, image_size=128, train_batch_size=64)

    wander_logger = wandb.WandbLogger(project='GANTest', log_model='all')
    wander_logger.watch(model)

    trainer = Trainer(
        logger=wander_logger,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=100,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback],
        enable_checkpointing=True
    )

    trainer.fit(model, dm, ckpt_path="Checkpoints/last.ckpt")