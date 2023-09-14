#!/usr/bin/env python3

import argparse
import sys

import argparse
import os
import torch
import torch.nn as nn
from torchvision import models

from dataloaders import Dataloaders
from trainer import Trainer
from utils import load_or_initialize_model, save_checkpoint, ensure_dir, get_device

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify flowers')
    parser.add_argument('data_dir', type=str, default='flowers', help='Directory containing the data')
    parser.add_argument('--save_dir', type=str, default='./', help='Path to the folder to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Pre-trained architecture',
                        choices=['vgg16', 'alexnet', 'resnet18'])
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Num of epochs')
    parser.add_argument('--test', action="store_true", help='Test the model instead of training')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training if available')

    return parser.parse_args()

def main():
    print('Starting training...')
    args = get_input_args()

    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir,'test')

    d = Dataloaders(train_dir, valid_dir, test_dir)
    image_datasets = d.get_image_datasets()
    dataloaders = d.get_dataloaders()
    device = get_device(args.gpu)

    ensure_dir(args.save_dir)
    checkpoint_path = os.path.join(args.save_dir, args.arch + '_checkpoint.pth')

    class_to_idx = image_datasets['train'].class_to_idx

    model, optimizer, start_epoch = load_or_initialize_model(checkpoint_path, args.arch, [args.hidden_units], args.learning_rate, device, class_to_idx)
    criterion = nn.NLLLoss()

    trainer = Trainer(model, criterion, dataloaders, device)

    if args.test:
        trainer.test()
    else:
        save_function = lambda trained_model, epoch: save_checkpoint(trained_model, checkpoint_path, args.arch, class_to_idx, optimizer, epoch)
        trainer.run(optimizer, args.epochs, save_function=save_function, save_interval=1, start_epoch=start_epoch)

if __name__ == "__main__":
    main()

