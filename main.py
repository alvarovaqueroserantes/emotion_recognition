import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.emotion_cnn import get_model
from utils.dataset import get_dataloaders
from utils.helpers import load_checkpoint_if_available, seed_everything
from utils.metrics import evaluate_model
from train import train_one_epoch
from test import test_model

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Cargar configuración
    config = load_config("configs/config.yaml")
    seed_everything(42)

    # Inicializar device
    device = torch.device("cuda" if (config["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # Preparar datos
    train_loader, val_loader = get_dataloaders(
        batch_size=config["batch_size"],
        input_size=config["input_size"]
    )

    # Crear modelo
    model = get_model(config["model_name"], config["num_classes"])
    model.to(device)

    # Definir pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Reanudar desde checkpoint si existe
    start_epoch = load_checkpoint_if_available(model, optimizer, config["checkpoint_path"])

    # Inicializar TensorBoard
    writer = SummaryWriter()

    # Entrenamiento
    for epoch in range(start_epoch, config["epochs"]):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer
        )

        # Evaluación en validación
        evaluate_model(model, val_loader, device, writer, epoch)

        # Guardar checkpoint
        os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, config["checkpoint_path"])

    writer.close()

    # Evaluación final
    print("Final Evaluation:")
    test_model(model, val_loader, device)

if __name__ == "__main__":
    main()
