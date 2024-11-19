# test.py
import torch
from torchvision import datasets, transforms, models

def load_data():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

def modify_alexnet():
    model = models.alexnet()
    model.classifier[6] = torch.nn.Linear(4096, 100)
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(model, test_loader):
    model.eval()
    correct_top1, correct_top5 = 0, 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = outputs.topk(5, dim=1)

            correct_top1 += torch.sum(predictions[:, 0] == labels).item()
            correct_top5 += torch.sum(predictions == labels.view(-1, 1)).sum().item()

    top1_accuracy = correct_top1 / len(test_loader.dataset)
    top5_accuracy = correct_top5 / len(test_loader.dataset)
    print(f'Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}')

def main():
    test_loader = load_data()
    model = modify_alexnet()

    # Load model weights (replace 'epoch5' or 'epoch20' as needed)
    model.load_state_dict(torch.load('model_epoch_5.pth'))

    test_model(model, test_loader)

if __name__ == "__main__":
    main()
