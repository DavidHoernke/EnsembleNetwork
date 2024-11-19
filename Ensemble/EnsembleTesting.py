import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18, alexnet, vgg16

# Define transforms for each model
resNet18transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
])

VGG16transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

alexNetTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets and data loaders
ResNetDataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=resNet18transform)
resNet18TestLoader = torch.utils.data.DataLoader(ResNetDataset, batch_size=32, shuffle=False)

AlexNetDataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=alexNetTransform)
alexNetTestLoader = torch.utils.data.DataLoader(AlexNetDataset, batch_size=32, shuffle=False)

VGG16Dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=VGG16transform)
VGG16TestLoader = torch.utils.data.DataLoader(VGG16Dataset, batch_size=32, shuffle=False)

TestLoaders = [resNet18TestLoader, alexNetTestLoader, VGG16TestLoader]

# Load models
resNet18Model = resnet18(pretrained=False)
AlexNetModel = alexnet(pretrained=False)
VGG16Model = vgg16(pretrained=False)

# Update final layers to match CIFAR-100 output
resNet18Model.fc = nn.Linear(resNet18Model.fc.in_features, 100)
AlexNetModel.classifier[6] = nn.Linear(4096, 100)
VGG16Model.classifier[6] = nn.Linear(4096, 100)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resNet18Model.to(device)
AlexNetModel.to(device)
VGG16Model.to(device)

# Load pre-trained model weights
VGG16Model.load_state_dict(torch.load('../VGG16/model_epoch_5.pth'))
AlexNetModel.load_state_dict(torch.load('../AlexNet/model_epoch_5.pth'))
resNet18Model.load_state_dict(torch.load('../Res18/model_epoch_5.pth'))

resNet18Model.eval()
AlexNetModel.eval()
VGG16Model.eval()

# Step 3: Evaluate ensemble predictions
correct_top1_avg = 0
correct_top1_max = 0
correct_top1_vote = 0
total = 0

with torch.no_grad():
    for (resNetInputs, alexNetInputs, VGG16Inputs) in zip(resNet18TestLoader, alexNetTestLoader, VGG16TestLoader):
        # Inputs and labels from the three datasets should match
        inputs_resnet, labels = resNetInputs
        inputs_alexnet, _ = alexNetInputs
        inputs_vgg, _ = VGG16Inputs

        inputs_resnet, inputs_alexnet, inputs_vgg, labels = inputs_resnet.to(device), inputs_alexnet.to(device), inputs_vgg.to(device), labels.to(device)

        # Get predictions from each model
        outputs_resnet = resNet18Model(inputs_resnet)
        outputs_alexnet = AlexNetModel(inputs_alexnet)
        outputs_vgg = VGG16Model(inputs_vgg)

        # Combine predictions for each method
        combined_outputs_avg = (outputs_resnet + outputs_alexnet + outputs_vgg) / 3
        combined_outputs_max, _ = torch.max(torch.stack([outputs_resnet, outputs_alexnet, outputs_vgg]), dim=0)

        predicted_resnet = torch.argmax(outputs_resnet, dim=1)
        predicted_alexnet = torch.argmax(outputs_alexnet, dim=1)
        predicted_vgg = torch.argmax(outputs_vgg, dim=1)
        majority_preds = torch.mode(torch.stack([predicted_resnet, predicted_alexnet, predicted_vgg]), dim=0).values

        # Calculate top-1 accuracy for average probability
        _, predicted_avg = torch.max(combined_outputs_avg, 1)
        correct_top1_avg += (predicted_avg == labels).sum().item()

        # Calculate top-1 accuracy for maximum probability
        _, predicted_max = torch.max(combined_outputs_max, 1)
        correct_top1_max += (predicted_max == labels).sum().item()

        # Calculate top-1 accuracy for majority voting
        correct_top1_vote += (majority_preds == labels).sum().item()

        total += labels.size(0)

# Calculate accuracies for each method
top1_accuracy_avg = 100 * correct_top1_avg / total
top1_accuracy_max = 100 * correct_top1_max / total
top1_accuracy_vote = 100 * correct_top1_vote / total

print(f'Ensemble accuracy on the CIFAR-100 test set (Top-1, Average Probability): {top1_accuracy_avg:.2f}%')
print(f'Ensemble accuracy on the CIFAR-100 test set (Top-1, Maximum Probability): {top1_accuracy_max:.2f}%')
print(f'Ensemble accuracy on the CIFAR-100 test set (Top-1, Majority Voting): {top1_accuracy_vote:.2f}%')