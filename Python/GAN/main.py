from GANDataHandler import GANDataModule
from GANModel import GAN
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import wandb
from GANConfig import GANConfig
import os

if __name__ == '__main__':

    training_params = GANConfig.training_params
    model_params = GANConfig.model_params
    data_module_params = GANConfig.data_module_params

    data_module_params.ds_source_path = 'Data/processed_anime_images'
    data_module_params.ds_enable_preprocess_images = False

    checkpoint_callback = ModelCheckpoint(dirpath='Checkpoints', filename='{epoch}-{train_steps}',
                                          auto_insert_metric_name=True, every_n_train_steps=500,
                                          save_last=True)

    model = GAN(model_params)

    if os.path.exists('Checkpoints/last.ckpt') and training_params.load_from_last_checkpoint:
        model = GAN.load_from_checkpoint('Checkpoints/last.ckpt')
    else:
        model.initialize_weights()

    dm = GANDataModule(data_module_params)

    if training_params.enable_logger:

        wander_logger = None

        wander_logger = wandb.WandbLogger(project='GANTest', log_model='all')
        wander_logger.watch(model)

        trainer = Trainer(
            logger=wander_logger,
            accelerator="gpu",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=20,
            callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback],
            enable_checkpointing=True
        )
    else:
        trainer = Trainer(
            accelerator="gpu",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=20,
            callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback],
            enable_checkpointing=True,

        )

    if os.path.exists('Checkpoints/last.ckpt') and training_params.load_from_last_checkpoint:
        trainer.fit(model, dm, ckpt_path='Checkpoints/last.ckpt')
    else:
        trainer.fit(model, dm)
