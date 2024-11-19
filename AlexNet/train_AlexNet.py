import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from torchvision import datasets, transforms, models
import time
import matplotlib.pyplot as plt
import copy

# Define a single transform for both training and validation (no data augmentation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((224, 224))
])

# Load CIFAR-100 dataset
full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# Split training data into training and validation sets
train_indices, val_indices = train_test_split(list(range(len(full_train_dataset))), test_size=0.2, random_state=42)

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

# Create data loaders for training and validation
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the model, loss function, optimizer, and scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.alexnet(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Use ReduceLROnPlateau for learning rate adjustment
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Track training history for plotting
train_losses = []
val_losses = []
epoch_durations = []

# Early stopping parameters
patience = 6
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

num_epochs = 30
total_start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Calculate validation loss after each epoch
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Save model at 5 epochs
    if epoch == 4:
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        print(f"Model saved at epoch {epoch + 1}")

    # Step the scheduler
    scheduler.step(val_loss)

    # Measure time for the epoch
    end_time = time.time()
    epoch_duration = end_time - start_time
    epoch_durations.append(epoch_duration)

    # Print epoch statistics and current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
          f"LR: {current_lr:.6f}, Duration: {epoch_duration:.2f} seconds")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        print(f"Best model updated at epoch {epoch + 1}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs with no improvement.")
        break

# Load the best model weights and save it
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'best_model.pth')
print("Best model saved.")

# Plot final training and validation losses and save the plot
plt.figure(figsize=(12, 5))

# Plot training and validation losses
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot epoch durations
total_time_seconds = sum(epoch_durations)
total_time_minutes = total_time_seconds / 60
total_time_hours = total_time_seconds / 3600

plt.subplot(1, 2, 2)
plt.plot(range(1, len(epoch_durations) + 1), epoch_durations, label='Epoch Duration (s)')
plt.xlabel('Epoch')
plt.ylabel('Duration (seconds)')
plt.title(
    f'Epoch Durations\nTotal Time: {total_time_seconds:.0f}s ({total_time_hours:.2f}h / {total_time_minutes:.2f}m)')
plt.tight_layout()

# Save the final plot
plt.savefig('training_plot.png')
plt.show()

total_end_time = time.time()
total_time = total_end_time - total_start_time
total_time_minutes = total_time / 60
total_time_hours = total_time / 3600
print(
    f"Total Training Time: {total_time:.2f} seconds ({total_time_hours:.2f} hours / {total_time_minutes:.2f} minutes)")

print("Training completed.")
