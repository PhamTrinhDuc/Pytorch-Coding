import argparse
import torchvision
from model import Trainer, create_mobilenet, create_retnest
from utils import Logger, AppPath, seed_everything
from config.catdog_config import CatDogArgs


LOGGER = Logger(name=__file__, log_file="training.log")
LOGGER.log.info("Start training ...")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_version", type=str, required=True, 
                        help="Version/directory to be used for training")
    parser.add_argument("--model_name", type=str, default="retnest18",
                        choices=["resnet18", "resnet34", "mobilenet_v2", "mobilenet_v3_small"], 
                        help="Model to be used for training")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Batch size for training")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="weight decay for optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate for optimizer")
    parser.add_argument("--best_model_metric", type=str, default="val_loss",
                        choices=["val_loss", "val_acc"], 
                        help="Metric for selecting the best model to logging to Mlflow")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device to be used for training")
    parser.add_argument("--seed", type=int, default=43,
                        help="Seed for reproducibility")
    parser.add_argument("--verbose", type=bool, default=True, 
                        help="logging info during training")
    
    args = parser.parse_args()
    seed_everything(args.seed)

    # ------------ create data
    data_path = AppPath.TRAIN_DATA_DIR/args.data_version
    try:
        assert data_path.exists()
    
    except:
        LOGGER.log.error(msg=f"data version: {args.data_version} not found")
        raise FileNotFoundError(f"Data version: {args.data_version} no found")
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path/"train",
        transform=CatDogArgs.train_transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=data_path/"val", 
        transform=CatDogArgs.test_transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path/"test",
        transform=CatDogArgs.test_transform
    )

    # -------------- create model 
    if args.model_name in ['mobilenet_v2', 'mobilenet_v3_small']:
        model = create_mobilenet(n_classes=CatDogArgs.n_classes, 
                                 model_name=args.model_name,
                                 load_pretrained=True)
    elif args.model_name in ["resnet18", "resnet34"]:
        model = create_retnest(n_classes=CatDogArgs.n_classes, 
                               model_name=args.model_name, 
                               load_pretrained=True)
    else:
        LOGGER.log.warning(msg="Invalid model. Choose in [resnet18, resnet34, mobilenet_v2, mobilenet_v3_small]")
    
    # ------------- create log mlflow
    mlflow_log_tags = {
        "data_version": args.data_version,
        "id2label": CatDogArgs.id2label,
        "label2id": CatDogArgs.label2id
    }
    mlflow_log_params = {
        "model": model.__class__.__name__, 
        "model_name": args.model_name,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size, 
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_model_metric": args.best_model_metric,
        "device": args.device,
        "seed": args.seed,
        "n_classes": CatDogArgs.n_classes,
        "image_size": CatDogArgs.img_size,
        "image_mean": CatDogArgs.mean,
        "image_std": CatDogArgs.std,
    }

    # -------------- training
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        mlflow_log_tags=mlflow_log_tags, 
        mlflow_log_params=mlflow_log_params, 
        best_model_metric=args.best_model_metric, 
        num_epochs=args.epochs,
        batch_size=args.batch_size, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device, 
        verbose=args.verbose,
    )
    trainer.train()
    LOGGER.log.info(f"Model Training Completed. Model: {args.model_name}, Data: {args.data_version}")

if __name__ == "__main__":
    main()
