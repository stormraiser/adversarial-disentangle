import torch
import torchvision

from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')

class ImageFolder(torchvision.datasets.DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=pil_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

    def get_nclass(self):
        return len(self.classes)

    def get_class_freq(self):
        class_count = torch.zeros(len(self.classes))
        for _, class_id in self.samples:
            class_count[class_id] += 1
        return class_count / class_count.sum().item()

    def get_class_to_id(self):
        return self.class_to_idx

class ParallelDataset(torch.utils.data.Dataset):

    def __init__(self, *datasets):
        self.length = len(datasets[0])
        for i in range(1, len(datasets)):
            if len(datasets[i]) != self.length:
                raise ValueError('Datasets does not have the same length!')
        self.datasets = datasets

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ret = []
        for dataset in self.datasets:
            ret.extend(dataset[index])
        return ret
