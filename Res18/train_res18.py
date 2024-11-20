import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
import argparse

# Argument parser for handling command-line arguments
argParser = argparse.ArgumentParser()
argParser.add_argument('-s', metavar='state', type=str, default="model.pt", help='Path to save model state (.pth)')
argParser.add_argument('-e', metavar='epochs', type=int, default=30, help='Number of epochs [default: 30]')
argParser.add_argument('-b', metavar='batch size', type=int, default=32, help='Batch size [default: 32]')

args = argParser.parse_args()

# Set parameters from arguments or use defaults
save_file = args.s
num_epochs = args.e
batch_size = args.b

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
cudnn.benchmark = True

# Define data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Load CIFAR-100 dataset
full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)

# Split indices for train and validation
indices = list(range(len(full_train_dataset)))
np.random.shuffle(indices)
train_size = int(0.9 * len(full_train_dataset))
train_indices, val_indices = indices[:train_size], indices[train_size:]

# Create subsets for training and validation
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

# Assign transforms
train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform_val

# Load test dataset
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)

# Define the ResNet-18 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Mixed precision training
scaler = GradScaler()

# Training settings
num_epochs = 200
best_acc = 0.0
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
total_start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = 100. * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    end_time = time.time()
    epoch_time = end_time - start_time

    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'LR: {optimizer.param_groups[0]["lr"]:.4f} '
          f'Train Loss: {train_loss:.4f} '
          f'Val Loss: {val_loss:.4f} '
          f'Train Acc: {train_acc:.2f}% '
          f'Val Acc: {val_acc:.2f}% '
          f'Time: {epoch_time:.2f}s')

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'bad models/best_resnet18_cifar100.pth')

    scheduler.step()

total_time = time.time() - total_start_time
print(f'Total Training Time: {total_time:.2f}s')

# Load best model weights
model.load_state_dict(best_model_wts)

# Plot training and validation losses
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# Plot training and validation accuracies
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Acc')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

print("Training complete.")

# Evaluate on test set
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with autocast():
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100. * test_correct / test_total
print(f'Test Accuracy: {test_acc:.2f}%')
