from GANDataHandler import GANDataModule
from DCGANModel import DCGAN
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, ModelSummary
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import wandb
from GANConfig import GANConfig
import os
import HelperUtils

if __name__ == '__main__':

    #HelperUtils.preprocess_images(source='Data/animefaces256cleaner', destination='Data/processed_anime_images', image_size=64)

    training_params = GANConfig.training_params
    model_params = GANConfig.model_params
    data_module_params = GANConfig.data_module_params

    checkpoint_callback = ModelCheckpoint(dirpath='Checkpoints', filename='{epoch}-{global_steps}',
                                          auto_insert_metric_name=True, every_n_train_steps=300,
                                          save_last=True)

    model = DCGAN(model_params)

    if os.path.exists(training_params.checkpoint_path) and training_params.load_from_last_checkpoint:
        print("\033[92mLoading model from checkpoint..")
        model = DCGAN.load_from_checkpoint(training_params.checkpoint_path)
        print("\033[92mFinished loading model from checkpoint..")
    else:
        print("\033[92mInitializing weights..")
        model.initialize_weights()

    dm = GANDataModule(data_module_params)

    if training_params.enable_logger:

        wander_logger = wandb.WandbLogger(project='GANTest', log_model='all')
        wander_logger.watch(model)
        print("\033[92mLogger Enabled..")

        trainer = Trainer(
            logger=wander_logger,
            accelerator="gpu",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=training_params.max_epochs,
            callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback],
            enable_checkpointing=True,
            benchmark=True
        )
    else:
        trainer = Trainer(
            accelerator="gpu",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=training_params.max_epochs,
            callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback, ModelSummary(max_depth=1)],
            enable_checkpointing=True,
            benchmark=True
        )

    if os.path.exists(training_params.checkpoint_path) and training_params.load_from_last_checkpoint:
        print("\033[92mTraining the model from checkpoint..")
        trainer.fit(model, dm, ckpt_path=training_params.checkpoint_path)
    else:
        print("\033[92mTraining the model..")
        trainer.fit(model, dm)

