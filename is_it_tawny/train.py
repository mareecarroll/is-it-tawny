#!/usr/bin/env python3
"""
train.py — Full training pipeline for Tawny vs Not‑Tawny classifier.

This script:
1. Loads labels.csv
2. Splits into train/validation sets
3. Builds a PyTorch Dataset + DataLoaders
4. Trains a ResNet18 model with transfer learning
5. Evaluates accuracy
6. Exports the model to ONNX for C++ inference
"""

import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ---------------------------------------------------------
#  Dataset Class
# ---------------------------------------------------------
class TawnyDataset(Dataset):
    """
    Custom dataset that:
    - Reads filenames + labels from a CSV
    - Loads images from disk
    - Applies transforms
    """

    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Map string labels → integers
        self.label_map = {"not_tawny": 0, "is_tawny": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.label_map[row["label"]]
        return img, label


# ---------------------------------------------------------
#  Main Training Function
# ---------------------------------------------------------
def main():

    # -----------------------------
    # Load CSV + Train/Val Split
    # -----------------------------
    df = pd.read_csv("labels.csv")

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)

    # -----------------------------
    # Image Transforms
    # -----------------------------
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # -----------------------------
    # Datasets + DataLoaders
    # -----------------------------
    train_ds = TawnyDataset("train.csv", "images", transform=train_tf)
    val_ds = TawnyDataset("val.csv", "images", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    # -----------------------------
    # Model Setup (ResNet18)
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final layer for 2‑class classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -----------------------------
    # Training Loop
    # -----------------------------
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss:.4f}")

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.3f}")

    # -----------------------------
    # Save PyTorch Model
    # -----------------------------
    torch.save(model.state_dict(), "tawny_classifier.pth")
    print("Saved PyTorch model → tawny_classifier.pth")

    # -----------------------------
    # Export to ONNX
    # -----------------------------
    dummy = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model,
        dummy,
        "tawny_classifier.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )

    print("Exported ONNX model → tawny_classifier.onnx")


# ---------------------------------------------------------
#  Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
