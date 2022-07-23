from CIFAR10DataLoader import CIFAR10Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, progress, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import os
import helper_utils as helper
from CIFARResnetModel import CIFAR10ResnetModel

if __name__ == '__main__':

    classes = [
        "airplane", "automobile", "bird", "cat",
        "deer", "dog", "frog", "horse", "ship", "truck",
    ]

    data_path = "Data"
    checkpoint_path = "Checkpoints"
    enable_logging = True
    max_epochs = 30
    wandb_logger = None

    if not os.path.exists(data_path):
        os.mkdir(path=data_path)

    if not os.path.exists(checkpoint_path):
        os.mkdir(path=checkpoint_path)

    dataset = CIFAR10Dataset(data_path, train_batch_size=64, val_batch_size=32,
                             test_batch_size=32, predict_batch_size=32)

    print(f"Creating an empty model...")
    model = CIFAR10ResnetModel(num_channel=3, num_classes=10,
                               learning_rate=0.0027, epsilon=0.1,
                               betas=(0.5, 0.999), weight_decay=0.001)

    model.initialize_weights()

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor="val_loss", save_top_k=2, save_last=True)
    TQDMProgress_callback = progress.TQDMProgressBar(refresh_rate=10, process_position=0)
    lr_callback = LearningRateMonitor()

    if enable_logging:
        wandb_logger = WandbLogger(project='CIFAR10Classification', log_model='all')
        wandb_logger.watch(model=model)

        trainer = Trainer(
            max_epochs=max_epochs, accelerator="gpu", devices=1,
            callbacks=[checkpoint_callback, TQDMProgress_callback, lr_callback],
            logger=wandb_logger,
            detect_anomaly=True
        )
    else:
        trainer = Trainer(
            max_epochs=max_epochs, accelerator="gpu", devices=1,
            callbacks=[checkpoint_callback, TQDMProgress_callback, lr_callback],
            detect_anomaly=True,
            fast_dev_run=True,
            # auto_lr_find=True
        )

    trainer.fit(model=model, datamodule=dataset, ckpt_path='Checkpoints/epoch=29-step=18750.ckpt')
    trainer.test(model=model, datamodule=dataset)

    # result = trainer.tuner.lr_find(model=model, datamodule=dataset)
    # print(result.suggestion())

    # Check how accuracy is for all classes
    dataset.setup(stage="predict")
    acc_d = helper.calculate_accuracy_for_each_class(num_classes=10, model_l=model,
                                                     pred_dl=dataset.predict_dataloader())
    helper.display_accuracy_chart(acc_d=acc_d, classes=classes)
