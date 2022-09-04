from Models.SRGANLightningModule import SRGANModel
from DataModules.SRGANLightningDataModule import SRGANDataModule
from pytorch_lightning import Trainer, seed_everything
import torch
import json
from Utils.HelperUtils import print_log, print_training_parameters, create_wandb_logger


if __name__ == '__main__':

    srgan_config_location = "../Configs/SRGAN_Config.json"

    print_log(text=f"Reading config data from : {srgan_config_location}")

    with(open(file=srgan_config_location, mode='r') as f):
        srgan_config = json.load(f)

    print_training_parameters(srgan_config)

    seed_everything(42)

    print_log(text="Creating SRGAN Model..")
    srgan_model = SRGANModel(
        gen_base_channels=srgan_config["Model"]["gen_base_channels"],
        gen_num_res_blocks=srgan_config["Model"]["gen_num_res_blocks"],
        gen_num_ps_blocks=srgan_config["Model"]["gen_num_ps_blocks"],
        disc_base_channels=srgan_config["Model"]["disc_base_channels"],
        disc_num_blocks=srgan_config["Model"]["disc_num_blocks"],
        lr_srresnet=srgan_config["Model"]["lr_srresnet"],
        lr_disc=srgan_config["Model"]["lr_disc"],
        lr_gen=srgan_config["Model"]["lr_gen"],
        gradient_accumulation_steps=srgan_config["Model"]["gradient_accumulation_steps"]
    )

    print_log(text=f"Is training SRResnet enabled : {srgan_config['Training']['enable_training_srresnet']}")
    srgan_model.enable_training_srresnet(enable=srgan_config['Training']['enable_training_srresnet'])

    print_log(text="Creating SRGAN Data Module..")
    srgan_data_module = SRGANDataModule(
        train_source_path=srgan_config["Data"]["train_source_path"],
        train_batch_size=srgan_config["Data"]["train_batch_size"],
        val_source_path=srgan_config["Data"]["val_source_path"],
        val_batch_size=srgan_config["Data"]["val_batch_size"],
        hr_image_size=srgan_config["Data"]["hr_image_size"],
        lr_image_size=srgan_config["Data"]["lr_image_size"]
    )

    wandb_logger = None

    if srgan_config["Training"]["enable_wandb_logger"]:
        wandb_logger = create_wandb_logger(project_name='SRGAN-Training')
        print_log(text="Wandb Logger Enabled..")
    else:
        print_log(text="Wandb Logger Disabled..")

    if srgan_config["Training"]["enable_training_srresnet"]:
        print_log(text="Creating Trainer for SRResnet..")
        trainer = Trainer(
            logger=wandb_logger if wandb_logger is not None else None,
            accelerator="gpu",
            devices=1 if torch.cuda.is_available() else None,
            enable_model_summary=True,
            enable_checkpointing=True,
            max_epochs=srgan_config["Training"]["srresnet_max_epochs"]
        )

        trainer.fit(model=srgan_model, datamodule=srgan_data_module)

    srgan_model.enable_training_srresnet(enable=False)

    print_log(text="Creating Trainer for SRGAN..")
    trainer = Trainer(
        logger=wandb_logger if wandb_logger is not None else None,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        enable_model_summary=True,
        enable_checkpointing=True,
        max_epochs=srgan_config["Training"]["srgan_max_epochs"]
    )

    trainer.fit(model=srgan_model, datamodule=srgan_data_module)
