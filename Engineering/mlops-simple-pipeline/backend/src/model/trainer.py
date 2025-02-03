import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import torch
import torch.nn as nn
from typing import Literal
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils.logger import Logger
load_dotenv()


LOGGER = Logger(name=__file__, log_file="training.log")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

try:
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
    LOGGER.log.info(f"MLFLOW TRACKING URI : {MLFLOW_TRACKING_URI}")
except Exception as e:
    LOGGER.log.info(f"Error: {str(e)}")
    LOGGER.log.error("set tracking uri and experiment failed")
    raise e


class Trainer: 
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 mlflow_log_tags: dict, 
                 mlflow_log_params: dict,
                 best_model_metric: Literal["val_loss", "val_acc"],
                 num_epochs: int,
                 batch_size: int,
                 lr: float,
                 weight_decay: float,
                 device: str='cpu', 
                 verbose: bool=False):
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(params=model.parameters(), 
                               lr=lr,
                               weight_decay=weight_decay)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.mlflow_log_tags = mlflow_log_tags
        self.mlflow_log_params = mlflow_log_params
        self.best_model_metric = best_model_metric
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def _create_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=self.batch_size, 
                                  shuffle=True,
                                  num_workers=2)
        val_loader = DataLoader(dataset=self.val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False, 
                                num_workers=2)
        return train_loader, val_loader
    
    def train(self):
        train_loader, val_loader = self._create_dataloader()
        run_name = f"{self.mlflow_log_params['model_name']}_{self.mlflow_log_tags['data_version']}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(tags=self.mlflow_log_tags)

            mlflow.log_params(params={
                "optimizer": self.optimizer.__class__.__name__,
                "criterion": self.criterion.__class__.__name__,
                **self.mlflow_log_params
            })

            best_val_loss = float("inf")
            best_val_acc  = float("-inf")
            best_val_loss_state_dict = None
            best_val_acc_state_dict = None

            for epoch in range(self.num_epochs):
                self.model.train()

                running_loss = 0.0
                running_acc = 0.0
                running_total = 0.0

                for i, (image, label) in enumerate(train_loader):
                    image, label = image.to(self.device), label.to(self.device)

                    output = self.model(image)
                    loss = self.criterion(output, label)

                    running_acc += (torch.argmax(output, dim=1) == label).sum().item()
                    running_loss += loss.item()
                    running_total += len(label)
                    
                    loss.backward()
                    self.optimizer.step()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = running_acc / running_total

            mlflow.log_metric(key="training_loss", value=f'{epoch_loss:.2f}', step=epoch)
            mlflow.log_metric(key="training_acc", value=f"{epoch_acc:.2f}", step=epoch)

            val_loss, val_acc = self.validate(data_loader=val_loader, epoch=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_acc_state_dict = self.model.state_dict()
            
            if val_loss < best_val_loss:
                val_loss = best_val_loss
                best_val_loss_state_dict = self.model.state_dict()
            
            if self.verbose:
                if epoch + 1 % 10 == 0:
                    LOGGER.log.info(
                        f"Epoch:\t[{epoch +1}/{self.num_epochs}] - Train Loss:\t{epoch_loss:.2f} - Val Loss:\t{val_loss:.2f}"
                    )
        
        mlflow.log_metric(key="best_val_loss", value=best_val_loss)
        mlflow.log_metric(key="best_val_acc", value=best_val_acc)

        if self.best_model_metric == "val_loss":
            best_model_state_dict = best_val_loss_state_dict
            LOGGER.log.info(f"Best model metric: {self.best_model_metric} - Best val loss: {best_val_loss:.4f}")

        elif self.best_model_metric == "val_acc":
            best_model_state_dict = best_val_acc_state_dict
            LOGGER.log.info(f"Best model metric: {self.best_model_metric} - Best acc loss: {best_val_acc:.4f}")

        else:
            raise ValueError(f"Invalid best_model_metric: {self.best_model_metric}")
    
        self.model.load_state_dict(best_model_state_dict)
        mlflow.pytorch.log_model(self.model, "model")
        
    def validate(self, data_loader: DataLoader, epoch: int, for_training: bool=True):
        self.model.eval()
        running_loss = 0.0
        running_acc  = 0.0
        running_total = 0.0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs)

                running_loss += loss.item()
                running_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
                running_total += len(labels)
            
            epoch_acc = running_acc / running_total
            epoch_loss = running_loss / len(data_loader)

            if for_training:
                mlflow.log_metric(key="validation_loss", value=f"{epoch_loss:.2f}", step=epoch)
                mlflow.log_metric(key="validation_acc", value=f"{epoch_acc:.2f}", step=epoch)
            
            return epoch_loss, epoch_acc

    def test(self):
        test_loader = DataLoader(dataset=self.val_dataset, 
                                 batch_size=self.batch_size,
                                 shuffle=False, 
                                 num_workers=2)
        return self.validate(data_loader=test_loader, epoch=0, for_training=False)

    def predict(self, image, transform, class_names):
        image = transform(image).unsqueeze(0) # transform + add batch dimension
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            _, predicted = torch.max(output, dim=1)
            predict_cls = class_names[predicted.item()]

            return predict_cls