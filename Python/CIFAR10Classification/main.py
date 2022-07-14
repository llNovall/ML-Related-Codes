from CIFAR10DataLoader import CIFAR10Dataset
from CIFAR10Model import CIFAR10Model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, progress
from pytorch_lightning.loggers import WandbLogger
import os
import helper_utils as helper

if __name__ == '__main__':

    classes = [
        "airplane", "automobile", "bird", "cat",
        "deer", "dog", "frog", "horse","ship", "truck",
    ]

    data_path = "Data"
    checkpoint_path = "Checkpoints"

    if not os.path.exists(data_path):
        os.mkdir(path=data_path)

    if not os.path.exists(checkpoint_path):
        os.mkdir(path=checkpoint_path)

    dataset = CIFAR10Dataset(data_path)

    print(f"Creating an empty model...")
    model = CIFAR10Model(num_channel=3, num_classes=10, learning_rate=0.001)
    model.initialize_weights()
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor="val_loss", save_top_k=2)
    TQDMProgress_callback = progress.TQDMProgressBar(refresh_rate=30, process_position=0)

    wandb_logger = WandbLogger(project='CIFAR10Classification', log_model='all')
    wandb_logger.watch(model=model)

    trainer = Trainer(
                      max_epochs=30, accelerator="gpu", devices=1,
                      callbacks=[checkpoint_callback, TQDMProgress_callback],
                      logger=wandb_logger,
                      detect_anomaly=True,
                      auto_lr_find=True
                     )

    trainer.fit(model=model, datamodule=dataset)
    trainer.test(model=model, datamodule=dataset)

    # result = trainer.tuner.lr_find(model=model, datamodule=dataset)
    # print(result.suggestion())

    # Check how accuracy is for all classes
    dataset.setup(stage="predict")
    acc_d = helper.calculate_accuracy_for_each_class(num_classes=10, model_l= model, pred_dl= dataset.predict_dataloader())
    helper.display_accuracy_chart(acc_d=acc_d, classes=classes)
