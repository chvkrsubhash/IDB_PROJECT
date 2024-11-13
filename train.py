import os
import imghdr
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing.image import load_img
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from PIL import Image

# # Kaggle setup
# files.upload()  # Upload kaggle.json file
# !kaggle datasets download -d msambare/fer2013
# !unzip fer2013.zip

image_exts = ['jpeg', 'jpg', 'png']

# Filter out non-image files
data_dir = 'train'
for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            if imghdr.what(file_path) not in image_exts:
                print(f'Removing non-image file: {file_path}')
                os.remove(file_path)
        except Exception as e:
            print(f'Issue with file {file_path}. Error: {e}')
            os.remove(file_path)

def count_files_in_subdirs(directory, set_name):
    counts = {item: len(os.listdir(os.path.join(directory, item)))
              for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))}
    return pd.DataFrame(counts, index=[set_name])

# Display training and testing file counts
train_count = count_files_in_subdirs('train', 'train')
test_count = count_files_in_subdirs('test', 'test')
print(train_count, test_count)
train_count.transpose().plot(kind='bar')
test_count.transpose().plot(kind='bar')

def plot_sample_images(directory, title, num_images=9):
    emotions = os.listdir(directory)
    plt.figure(figsize=(15, 10))
    for i, emotion in enumerate(emotions, 1):
        folder = os.path.join(directory, emotion)
        img_path = os.path.join(folder, os.listdir(folder)[42])
        img = plt.imread(img_path)
        plt.subplot(3, 4, i)
        plt.imshow(img, cmap='gray')
        plt.title(emotion)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

plot_sample_images('train', 'Sample Images from Each Emotion Class')

# Data transformations for PyTorch
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
}

# Load datasets and create data loaders
train_dataset = datasets.ImageFolder('train', transform=data_transforms['train'])
test_dataset = datasets.ImageFolder('test', transform=data_transforms['test'])
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 125
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Define CNN + RNN model
class EmotionRNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionRNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25),
            nn.Conv2d(128, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25),
        )
        self.rnn = nn.RNN(512*3*3, 128, num_layers=2, batch_first=True, dropout=0.25)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256), nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1).unsqueeze(1)
        x, _ = self.rnn(x)
        return self.classifier(x[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionRNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training loop with early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=10, min_delta=0.01)

def train_model(model, train_loader, val_loader, num_epochs=100):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / total
        val_loss, val_correct, val_total = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return train_losses, val_losses

train_losses, val_losses = train_model(model, train_loader, val_loader)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.show()
