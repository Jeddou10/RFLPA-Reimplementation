import torch
import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(dataset_name: str, num_classes: int = 10):
    """Get appropriate model based on dataset"""
    if dataset_name in ['mnist', 'fashion-mnist']:
        # Simple CNN for MNIST dataset
        return SimpleCNN(input_channels=1, num_classes=num_classes)
    elif dataset_name == 'cifar10':
        # CIFAR-10 dataset
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    elif dataset_name == 'cifar100':
        # CIFAR-100 dataset
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")