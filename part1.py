
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

train_df = pd.read_csv('C:/Users/katum/Documents/SPRING2025/STATISTICAL LEARNING/Final Project/train.csv')
test_df = pd.read_csv('C:/Users/katum/Documents/SPRING2025/STATISTICAL LEARNING/Final Project/test.csv')

class DigitDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.is_test = is_test
        if not is_test:
            self.labels = df['label'].values
            self.images = df.drop('label', axis=1).values
        else:
            self.images = df.values
            self.labels = None

        self.images = self.images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

        # Normalize using MNIST dataset mean/std
        self.transform = transforms.Normalize((0.1307,), (0.3081,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        image = self.transform(image)  
        if self.is_test:
            return image
        label = torch.tensor(self.labels[idx]).long()
        return image, label

full_dataset = DigitDataset(train_df)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   
        x = self.pool(F.relu(self.conv2(x)))   
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))  
        return self.fc2(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20): 
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

test_dataset = DigitDataset(test_df, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=64)

all_preds = []
model.eval()
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

submission = pd.DataFrame({
    'ImageId': np.arange(1, len(all_preds)+1),
    'Label': all_preds
})
submission.to_csv('C:/Users/katum/Documents/SPRING2025/STATISTICAL LEARNING/Final Project/kaggle_sub.csv', index=False)
