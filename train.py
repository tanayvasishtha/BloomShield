import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import random

class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data():
    train_images = glob.glob('data/train/*.jpg') + glob.glob('data/train/*.JPEG')
    valid_images = glob.glob('data/valid/*.jpg') + glob.glob('data/valid/*.JPEG')
    test_images = glob.glob('data/test/*.jpg') + glob.glob('data/test/*.JPEG')
    train_labels, valid_labels, test_labels = [], [], []
    class_to_idx = {}
    idx = 0
    def process_images(images, labels):
        for img in images:
            base = os.path.basename(img)
            parts = base.split('_', 1)
            if len(parts) < 2:
                continue
            class_name = parts[0]
            if class_name not in class_to_idx:
                class_to_idx[class_name] = idx
                idx += 1
            label = class_to_idx[class_name]
            labels.append(label)
        return images, labels
    train_images, train_labels = process_images(train_images, train_labels)
    valid_images, valid_labels = process_images(valid_images, valid_labels)
    test_images, test_labels = process_images(test_images, test_labels)
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels, class_to_idx

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Assuming 224x224 input
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels, class_to_idx = load_data()
    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_images)}, Valid: {len(valid_images)}, Test: {len(test_images)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PlantDataset(train_images, train_labels, transform)
    valid_dataset = PlantDataset(valid_images, valid_labels, transform)
    test_dataset = PlantDataset(test_images, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SimpleCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(8):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total} %")

    torch.save(model.state_dict(), 'bloomshield_model.pth')
    print("Model saved as bloomshield_model.pth") 