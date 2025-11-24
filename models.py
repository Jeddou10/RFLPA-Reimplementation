import torch
import torch.nn as nn
import torchvision.models as models

def get_model(dataset_name: str, num_classes: int = 10):
    """Get appropriate model based on dataset"""
    if dataset_name in ['mnist', 'fashion-mnist']:
        # Simple CNN for MNIST dataset
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    elif dataset_name == 'cifar10':
        #cifar10 dataseet
        model = models.resnet9(num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  
        return model
    elif dataset_name == 'cifar100':
        #cifar100 dataset
        model = models.resnet9(num_classes=100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")