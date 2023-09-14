import os

import torch
from torch import nn, optim
from torchvision import models
from network import Network
from typing import Tuple, List

def load_or_initialize_model(checkpoint_path: str, arch: str, hidden_units: List[int], learning_rate: float, device: torch.device, class_to_idx: dict) -> Tuple[nn.Module, optim.Optimizer, int]:
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f'Checkpoint {checkpoint_path} found. Loading model from checkpoint.')
        model, optimizer, start_epoch = load_checkpoint(checkpoint_path, device)
    else:
        print('Checkpoint not found. Initializing model with new classifier.')
        model_func = getattr(models, arch)
        model = model_func(pretrained=True)

        num_of_features = model.classifier[0].in_features
        num_of_classes = len(class_to_idx)
        for p in model.parameters():
            p.requires_grad = False

        classifier = Network(num_of_features, num_of_classes, hidden_units)

        for p in model.parameters():
            p.requires_grad = True

        model.classifier = classifier
        model = model.to(device)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, optimizer, start_epoch

def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, optim.Optimizer, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_func = getattr(models, checkpoint['arch'])
    model = model_func(pretrained=False)

    num_of_features = checkpoint['num_of_features']
    num_of_classes = checkpoint['num_of_classes']
    hidden_units = checkpoint['hidden_units']
    classifier = Network(num_of_features, num_of_classes, hidden_units)
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    model = model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    return model, optimizer, start_epoch

def save_checkpoint(model: nn.Module, filepath: str, arch: str, class_to_idx: dict, optimizer: optim.Optimizer, epoch: int) -> None:
    learning_rate = 0

    if len(optimizer.param_groups) > 0:
        learning_rate = optimizer.param_groups[0]['lr']

    checkpoint = {
        'num_of_features': model.classifier.num_of_features,
        'num_of_classes': len(class_to_idx),
        'arch': arch,
        'hidden_units': model.classifier.hidden_units,
        'class_to_idx': class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'learning_rate': learning_rate,
        'epoch': epoch,
    }
    torch.save(checkpoint, filepath)

def ensure_dir(file_path: str) -> None:
    if os.path.exists(file_path):
        return
    os.makedirs(file_path)

def get_device(gpu: bool) -> torch.device:
    return torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

