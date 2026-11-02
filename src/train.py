import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os


def train_model(model, train_loader, val_loader, device, epochs=10, save_path='best_model.pth'):
    model = model.to(device)

    # Class weights (Normal: 234, Pneumonia: 390)
    class_weights = [1.0 / 234, 1.0 / 390]
    class_weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

        # --- Validation Accuracy ---
        if val_loader:
            val_acc = validate_model(model, val_loader, device)
            print(f"Validation Accuracy: {val_acc * 100:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print("✅ Best model saved.")



def validate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc
