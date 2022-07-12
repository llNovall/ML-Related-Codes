from MNISTDataLoader import MNISTDataset
from MNISTModel import MNISTModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    wandb_logger = WandbLogger(project='MNISTClassification', log_model='all')
    dataset = MNISTDataset()
    model = MNISTModel(learning_rate=0.005754399373371567)

    trainer = Trainer(max_epochs=10, accelerator="gpu", devices=1, logger=wandb_logger)

    trainer.fit(model=model, datamodule=dataset)
    trainer.test(model=model, datamodule=dataset)
