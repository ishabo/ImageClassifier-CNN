import os
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class Dataloaders:

    def __init__(self, train_dir: str, valid_dir: str, test_dir: str, batch_size: int = 32, shuffle: bool = True):
        self.train_dir = self.validate_and_get_full_dir_path(train_dir)
        self.valid_dir = self.validate_and_get_full_dir_path(valid_dir)
        self.test_dir = self.validate_and_get_full_dir_path(test_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_size = 224
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.data_transforms = self.get_data_transforms()

    def validate_and_get_full_dir_path(self, dir_path: str) -> str:
        if not os.path.isdir(dir_path):
            raise ValueError('Directory not found: {}'.format(dir_path))

        return os.path.abspath(dir_path)

    def get_data_transforms(self) -> dict:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),

            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
            ]),
        }

        self.data_transforms = data_transforms
        return data_transforms

    def get_image_datasets(self) -> dict:
        image_datasets = {
            'train': ImageFolder(self.train_dir, transform=self.data_transforms['train']),
            'valid': ImageFolder(self.valid_dir, transform=self.data_transforms['valid']),
            'test': ImageFolder(self.test_dir, transform=self.data_transforms['test']),
        }

        self.image_datasets = image_datasets
        return image_datasets

    def get_dataloaders(self) -> dict:
        image_datasets = self.get_image_datasets()

        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=self.batch_size, shuffle=self.shuffle),
            'valid': DataLoader(image_datasets['valid'], batch_size=self.batch_size, shuffle=False),
            'test': DataLoader(image_datasets['test'], batch_size=self.batch_size, shuffle=False),
        }

        return dataloaders
