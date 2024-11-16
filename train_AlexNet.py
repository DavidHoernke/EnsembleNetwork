# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import os


def load_data():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load the CIFAR-100 dataset
    full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader

def modify_alexnet():
    model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 100)
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    train_loss_history = []
    val_loss_history = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        
        # Save model after 5 epochs and after full convergence
        if epoch + 1 == 5 or epoch + 1 == num_epochs:
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), f'saved_models/alexnet_epoch{epoch+1}.pth')

    return train_loss_history, val_loss_history

def plot_loss_curves(train_loss, val_loss):
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('AlexNet Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('alexnet_loss_curve.png')
    plt.show()

def main():
    train_loader, val_loader = load_data()
    model = modify_alexnet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer)
    plot_loss_curves(train_loss, val_loss)

if __name__ == "__main__":
    main()
