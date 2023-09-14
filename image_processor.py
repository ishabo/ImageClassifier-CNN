import numpy as np
from PIL import Image

def crop_image(img, crop_dim=224):
    ''' Crop the input PIL image to a square of side crop_dim pixels. '''

    left_margin = (img.width-crop_dim)/2
    bottom_margin = (img.height-crop_dim)/2
    right_margin = left_margin + crop_dim
    top_margin = bottom_margin + crop_dim
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    return img

def resize_image(img, size=256):
    ''' Resize the input PIL image to a square of side size pixels. '''

    return img.resize((size, int(size*(img.height/img.width))) if img.width < img.height else (int(size*(img.width/img.height)), size))

def normalize_image(img, means, stds):
    img = np.array(img)/255
    img = (img - means)/stds
    return img

def process_image(image, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img = Image.open(image)
    img = img.resize((256, int(256*(img.height/img.width))) if img.width < img.height else (int(256*(img.width/img.height)), 256))
    img = crop_image(img)
    img = normalize_image(img, means, stds)
    img = img.transpose((2, 0, 1))

    return img
