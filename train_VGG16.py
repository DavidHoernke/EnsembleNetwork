import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
import matplotlib.pyplot as plt

# Step 1: Load CIFAR-100 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

# Split the dataset
train_subset, val_subset = random_split(dataset, [train_size, val_size])

#EVERYTHING ABOVE THIS WILL BE SAME IN ALL TRAIN FILES

trainloader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
valloader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)

# Step 2: Load VGG16 model and modify for CIFAR-100
model = vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 100)  # Adjust for CIFAR-100 output classes
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Step 3: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Step 4: Training loop
def train(model, epochs, save_checkpoint_after=None):
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        train_loss_list.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(valloader)
        val_loss_list.append(val_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save checkpoint after specified epochs
        if save_checkpoint_after and (epoch + 1) == save_checkpoint_after:
            torch.save(model.state_dict(), f'vgg16_checkpoint_epoch_{save_checkpoint_after}.pth')

    # Save final model state
    torch.save(model.state_dict(), 'vgg16_final_model.pth')

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_plot.png')
    plt.show()


# Step 5: Train model
train(model, epochs=50, save_checkpoint_after=5)
