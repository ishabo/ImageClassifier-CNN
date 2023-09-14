#!/usr/bin/env python3

import argparse
import json
import torch

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
from collections import OrderedDict
from PIL import Image
from typing import List, Tuple

from image_processor import process_image
from utils import load_checkpoint, get_device

def plot_probability(top_flowers: List[str], top_probs: List[float]) -> None:
    ''' Plot the probabilities of the top x flowers '''

    _, ax = plt.subplots()
    y_pos = np.arange(len(top_flowers))
    ax.barh(y_pos, top_probs, align='center', color='blue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_flowers)
    ax.invert_yaxis()
    ax.set_xlabel('Probs')
    plt.show()

def predict(image_path: str, cat_to_name: dict, model: nn.Module, topk: int = 5) -> Tuple[List[float], List[int], List[str]]:
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    logps = model.forward(image)
    ps = torch.exp(logps)
    top_probs, top_labels = ps.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labels = top_labels.detach().numpy().tolist()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]
    top_flowers = [cat_to_name[label] for label in top_labels]

    return top_probs, top_labels, top_flowers

def get_input_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='the path to image to predict')
    parser.add_argument('checkpoint_path', type=str, help='the path to a pre-trained checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='top K for most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='for mapping labels to categories')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training if available')
    parser.add_argument('--plot', action='store_true', help='plots prediction')

    return parser.parse_args()

def main() -> None:
    print("Predicting flower name and class...\n")
    args = get_input_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    device = get_device(args.gpu)
    model, _, _ = load_checkpoint(args.checkpoint_path, device)
    top_probs, _, top_flowers = predict(args.image_path, cat_to_name, model, topk=args.top_k)

    if args.plot:
        plot_probability(top_flowers, top_probs)

    print(top_probs, top_flowers)
    print(f'Flower name: {top_flowers[0]}')
    print(f'Probability: {top_probs[0]}')

if __name__ == "__main__":
    main()
