import random

import numpy as np
import torchvision.transforms
import os

import torch.utils.data
from torchvision.io import read_image


def yolobbox2bbox(x, y, w, h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2


resize = torchvision.transforms.Resize(640)


class YOLODataset(torch.utils.data.Dataset):

    def __init__(self, image_path, annotation_path, label_file, transform=None, target_transforms=None, shuffle=False):

        self.image_path = image_path
        self.annotation_path = annotation_path
        self.transform = transform
        self.target_transforms = target_transforms

        self.image_files = os.listdir(image_path)
        if shuffle:
            random.shuffle(self.image_files)
        self.ids = [i for i in range(len(os.listdir(image_path)))]

        self.annotation_files = [''.join(img.split('.')[:-1]) + '.txt' for img in self.image_files]

        with open(label_file, 'r') as f:
            classes = f.readlines()
            classes = [c.strip() for c in classes]
            self.class_names = classes

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = read_image(os.path.join(self.image_path, self.image_files[image_id]))
        if image.shape[0] > 3:
            image = image[:3, :, :]
        image = image.float()
        size = image.size()[1:3]
        image = resize(image)

        x_scale = size[0] / image.size()[1]
        y_scale = size[1] / image.size()[2]

        # Update bounding box coordinates
        boxes = [bb * torch.tensor([x_scale, y_scale, x_scale, y_scale]) for bb in torch.tensor(boxes)]
        print(torch.stack(boxes))
        return image, torch.stack(boxes), labels

    def __len__(self):
        return len(self.ids)