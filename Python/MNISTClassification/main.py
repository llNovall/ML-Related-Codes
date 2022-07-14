from MNISTDataLoader import MNISTDataset
from MNISTModel import MNISTModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import os


def calculate_accuracy_for_each_class(model_l, pred_dl):
    acc_dict = {"0": 0.0,
                "1": 0.0,
                "2": 0.0,
                "3": 0.0,
                "4": 0.0,
                "5": 0.0,
                "6": 0.0,
                "7": 0.0,
                "8": 0.0,
                "9": 0.0}

    total_dict = {}

    final_dict = {}
    dataset.setup(stage="predict")

    with torch.no_grad():

        for batch in pred_dl:

            x, y = batch
            y_hat = model_l(x)
            y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)

            for i in range(len(y_hat)):
                if y_hat[i] == y[i]:
                    acc_dict[f"{y[i]}"] = acc_dict.get(f"{y[i]}", 0) + 1

                total_dict[f"{y[i]}"] = total_dict.get(f"{y[i]}", 0) + 1

        for key, value in acc_dict.items():
            total = total_dict[key]

            final_dict[key] = value / total

    return final_dict


if __name__ == '__main__':

    wandb_logger = WandbLogger(project='MNISTClassification', log_model='all')
    dataset = MNISTDataset()

    print(f"Creating an empty model...")
    model = MNISTModel(num_channel=1, learning_rate=0.005)

    checkpoint_path = "Checkpoints"
    if not os.path.exists(checkpoint_path):
        os.mkdir(path=checkpoint_path)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor="val_loss", save_top_k=2)

    trainer = Trainer(max_epochs=3, accelerator="gpu", devices=1, callbacks=[checkpoint_callback],
                      progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer.fit(model=model, datamodule=dataset)
    trainer.test(model=model, datamodule=dataset)

    # Check how accuracy is for all classes
    dataset.setup(stage="predict")
    acc_d = calculate_accuracy_for_each_class(model, dataset.predict_dataloader())

    print(f"Accuracy Chart")
    print(f"--------------")
    for key, value in acc_d.items():
        print(f"{key} : {(value * 100):.2f} %")
