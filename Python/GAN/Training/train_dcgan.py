from Models.DCGANLightningModule import DCGANModel
from DataModules.DCGANLightningDataModule import DCGANDataModule
from pytorch_lightning import Trainer, seed_everything
import torch
import json
from Utils.HelperUtils import print_log, print_training_parameters, create_wandb_logger


if __name__ == '__main__':

    seed_everything(42)

    dcgan_config_location = "../Configs/DCGAN_Config.json"

    print_log(text=f"Reading config data from : {dcgan_config_location}")

    with(open(file=dcgan_config_location, mode='r') as f):
        dcgan_config = json.load(f)

    print_training_parameters(dcgan_config)

    print_log(text="Creating DCGAN Model..")

    dcgan_model = DCGANModel(
        latent_dim=dcgan_config["Model"]["latent_dim"],
        gen_base_channels=dcgan_config["Model"]["gen_base_channels"],
        disc_base_channels=dcgan_config["Model"]["disc_base_channels"],
        lr_disc=dcgan_config["Model"]["lr_disc"],
        lr_gen=dcgan_config["Model"]["lr_gen"],
        gradient_accumulation_steps=dcgan_config["Model"]["gradient_accumulation_steps"],
        lr_scheduler_steps=dcgan_config["Model"]["lr_scheduler_steps"],
        critic_lambda=dcgan_config["Model"]["critic_lambda"],
        critic_opt_repeat=dcgan_config["Model"]["critic_opt_repeat"],
        loss=dcgan_config["Model"]["loss"],
        num_blocks=dcgan_config["Model"]["num_blocks"],
        block_type=dcgan_config["Model"]["block_type"],
    )

    dcgan_model.initialize_weights()

    print_log(text="Creating DCGAN Data Module..")
    dcgan_data_module = DCGANDataModule(
        train_source_path=dcgan_config["Data"]["train_source_path"],
        train_batch_size=dcgan_config["Data"]["train_batch_size"],
        image_size=dcgan_config["Data"]["image_size"]
    )

    wandb_logger = None

    if dcgan_config["Training"]["enable_wandb_logger"]:
        wandb_logger = create_wandb_logger(project_name='DCGAN-Training')
        print_log(text="Wandb Logger Enabled..")
    else:
        print_log(text="Wandb Logger Disabled..")

    print_log(text="Creating Trainer for DCGAN..")

    trainer = Trainer(
        logger=wandb_logger if wandb_logger is not None else None,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        enable_model_summary=True,
        enable_checkpointing=True,
        max_epochs=dcgan_config["Training"]["max_epochs"]
    )

    trainer.fit(model=dcgan_model, datamodule=dcgan_data_module)
