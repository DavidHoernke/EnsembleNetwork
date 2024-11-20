# test.py
import torch
from torchvision import datasets, transforms, models
import argparse

parser = argparse.ArgumentParser(description="Test AlexNet on CIFAR-100")
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the model checkpoint file (.pth)")
args = parser.parse_args()

import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

# Argument parser to handle command-line arguments
parser = argparse.ArgumentParser(description="Test AlexNet on CIFAR-100")
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the model checkpoint file (.pth)")
args = parser.parse_args()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.alexnet()
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 100)  # Adjust output layer for 100 classes
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device)
model.eval()

# Prepare the CIFAR-100 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the model
correct_top1 = 0
correct_top5 = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted_top5 = outputs.topk(5, dim=1)
        total += targets.size(0)
        correct_top1 += (predicted_top5[:, 0] == targets).sum().item()
        correct_top5 += sum([targets[i] in predicted_top5[i] for i in range(targets.size(0))])

top1_accuracy = 100 * correct_top1 / total
top5_accuracy = 100 * correct_top5 / total

print(f"Accuracy of the model on the CIFAR-100 test set (Top-1): {top1_accuracy:.2f}%")
print(f"Accuracy of the model on the CIFAR-100 test set (Top-5): {top5_accuracy:.2f}%")

# Example predictions
classes = test_dataset.classes
for inputs, targets in test_loader:
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, predicted_top5 = outputs.topk(5, dim=1)
    for i in range(5):  # Show 5 examples
        print(f"Actual: {classes[targets[i]]}, Predicted Top-5: {[classes[p] for p in predicted_top5[i]]}")
    break


# def load_data():
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
#     return test_loader

# def modify_alexnet():
#     model = models.alexnet()
#     model.classifier[6] = torch.nn.Linear(4096, 100)
#     return model.to('cuda' if torch.cuda.is_available() else 'cpu')

# def test_model(model, test_loader):
#     model.eval()
#     correct_top1, correct_top5 = 0, 0
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predictions = outputs.topk(5, dim=1)

#             correct_top1 += torch.sum(predictions[:, 0] == labels).item()
#             correct_top5 += torch.sum(predictions == labels.view(-1, 1)).sum().item()

#     top1_accuracy = correct_top1 / len(test_loader.dataset)
#     top5_accuracy = correct_top5 / len(test_loader.dataset)
#     print(f'Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}')

# def main():
#     test_loader = load_data()
#     model = modify_alexnet()

#     # Load model weights (replace 'epoch5' or 'epoch20' as needed)
#     model.load_state_dict(torch.load('model_epoch_5.pth'))

#     test_model(model, test_loader)

# if __name__ == "__main__":
#     main()
