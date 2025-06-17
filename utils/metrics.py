import torch
from sklearn.metrics import f1_score, accuracy_score

def evaluate_model(model, dataloader, device, writer, epoch):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"[Validation] Epoch {epoch+1}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

    writer.add_scalar("Accuracy/val", acc * 100, epoch)
    writer.add_scalar("F1_score/val", f1, epoch)

    return acc, f1
