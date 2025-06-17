import torch.nn as nn
import torchvision.models as models

def get_model(model_name: str, num_classes: int):
    """
    Devuelve un modelo preentrenado con la capa final adaptada.

    Args:
        model_name (str): Nombre del modelo base (e.g., "resnet18").
        num_classes (int): Número de clases de salida (e.g., 7 para emociones).

    Returns:
        model (nn.Module): Modelo de PyTorch listo para entrenamiento.
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

    return model
