import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import vgg16

# Step 1: Load the CIFAR-100 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Load the trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 100)  # Adjust for CIFAR-100 output classes
model = model.to(device)

# Load the saved model checkpoint (update the path to your checkpoint)
checkpoint_path = 'model_epoch_5.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Step 3: Evaluate the model for top-1 and top-5 accuracy
correct_top1 = 0
correct_top5 = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Calculate top-1 accuracy
        total += labels.size(0)
        correct_top1 += (predicted == labels).sum().item()

        # Calculate top-5 accuracy
        top5_preds = torch.topk(outputs, 5, dim=1).indices
        for i in range(labels.size(0)):
            if labels[i] in top5_preds[i]:
                correct_top5 += 1

# Calculate accuracies
top1_accuracy = 100 * correct_top1 / total
top5_accuracy = 100 * correct_top5 / total
print(f'Accuracy of the model on the CIFAR-100 test set (Top-1): {top1_accuracy:.2f}%')
print(f'Accuracy of the model on the CIFAR-100 test set (Top-5): {top5_accuracy:.2f}%')

# Step 4: Optional - Print some test predictions
classes = test_dataset.classes  # CIFAR-100 class names
data_iter = iter(testloader)
images, labels = next(data_iter)
images, labels = images.to(device), labels.to(device)

outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('\nSample predictions:')
for i in range(5):  # Print 5 sample predictions
    top5 = torch.topk(outputs[i], 5).indices
    top5_classes = [classes[idx] for idx in top5]
    print(f'Actual: {classes[labels[i]]}, Predicted Top-5: {top5_classes}')
