import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer: SummaryWriter):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"[Train] Epoch {epoch+1}", leave=False)

    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = 100 * correct / total
        progress.set_postfix(loss=loss.item(), acc=acc)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)
