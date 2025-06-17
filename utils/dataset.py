import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(input_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

def get_dataloaders(data_dir="data", batch_size=64, input_size=224):
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "test")  # Puede renombrarse como "val" si prefieres

    train_dataset = datasets.ImageFolder(train_path, transform=get_transforms(input_size, is_train=True))
    val_dataset = datasets.ImageFolder(val_path, transform=get_transforms(input_size, is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
