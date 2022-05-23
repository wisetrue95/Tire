from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


class trainDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle, num_workers=1):

        self.trsfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(
            data_dir,
            transform=self.trsfm
        )

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': True
        }
        super().__init__(dataset, **self.init_kwargs)



class valDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle, num_workers=1):
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self.trsfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(
            data_dir,
            transform=self.trsfm
        )
        super().__init__(dataset, **self.init_kwargs)



class testDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle, num_workers=1):
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self.trsfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(
            data_dir,
            transform=self.trsfm
        )
        super().__init__(dataset, **self.init_kwargs)


class attentionDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle, num_workers=1):
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self.trsfm = transforms.Compose([
            transforms.Resize((7,7)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(
            data_dir,
            transform=self.trsfm
        )
        super().__init__(dataset, **self.init_kwargs)


class attention_valDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle, num_workers=1):
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self.trsfm = transforms.Compose([
            transforms.Resize((7,7)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(
            data_dir,
            transform=self.trsfm
        )
        super().__init__(dataset, **self.init_kwargs)