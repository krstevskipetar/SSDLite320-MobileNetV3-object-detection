import pdb
import random
from typing import Sequence

import numpy as np
import torchvision.transforms
import os
from os.path import exists, join

import torch.utils.data
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


def yolobbox2bbox(bbox: Sequence[int]):
    x, y, w, h = bbox
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return int(x1), int(y1), int(x2), int(y2)


def scale_bbox(bbox: Sequence[int], image_width: int, image_height: int):
    x, y, w, h = bbox
    x *= image_width
    w *= image_width
    y *= image_height
    h *= image_height
    return x, y, w, h


resize = torchvision.transforms.Resize((640, 640), antialias=True)


def normalized_to_absolute(bbox, image_width, image_height):
    """
    Convert normalized bounding box coordinates to absolute coordinates.

    Parameters:
    - bbox: Tuple (x, y, w, h) representing normalized coordinates.
    - image_width: Width of the image.
    - image_height: Height of the image.

    Returns:
    - Tuple (x_min, y_min, x_max, y_max) representing absolute coordinates.
    """
    x, y, w, h = bbox
    x_min = int(x * image_width)
    y_min = int(y * image_height)
    x_max = int((x + w) * image_width)
    y_max = int((y + h) * image_height)
    return x_min, y_min, x_max, y_max


def resize_bbox(bbox, original_size, new_size):
    """
    Resize bounding box coordinates to a new image size.

    Parameters:
    - bbox: Tuple (x_min, y_min, x_max, y_max) representing absolute coordinates.
    - original_size: Tuple (original_width, original_height) representing the original image size.
    - new_size: Tuple (new_width, new_height) representing the new image size.

    Returns:
    - Tuple (resized_x_min, resized_y_min, resized_x_max, resized_y_max) representing resized coordinates.
    """
    x_min, y_min, x_max, y_max = bbox
    original_width, original_height = original_size
    new_width, new_height = new_size

    resized_x_min = int(x_min * (new_width / original_width))
    resized_y_min = int(y_min * (new_height / original_height))
    resized_x_max = int(x_max * (new_width / original_width))
    resized_y_max = int(y_max * (new_height / original_height))

    return resized_x_min, resized_y_min, resized_x_max, resized_y_max


class YOLODataset(torch.utils.data.Dataset):

    # todo: add input validation
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
        self._validate_input()
        with open(label_file, 'r') as f:
            classes = f.readlines()
            classes = [c.strip() for c in classes]
            self.class_names = classes

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def _validate_input(self):
        i = 0
        for idx, (image_file, annotation_file) in enumerate(zip(self.image_files, self.annotation_files)):
            if not exists(join(self.image_path, image_file)) or not exists(join(self.image_path, image_file)):
                self.image_files.pop(idx)
                self.annotation_files.pop(idx)
                i += 1
            try:
                boxes, labels = self._get_annotation(idx)
                if np.shape(boxes)[1] > 4:
                    self.annotation_files.pop(idx)
                    self.image_files.pop(idx)
                    i += 1
            except:
                self.image_files.pop(idx)
                self.annotation_files.pop(idx)
                i += 1

        print(f"Removed {i} invalid samples")

    def _get_annotation(self, image_id):
        with open(os.path.join(self.annotation_path, self.annotation_files[image_id]), 'r') as f:
            ann_file = f.readlines()
            ann_file = [[float(a) for a in ann_line.strip().split(' ')] for ann_line in ann_file]

        ann_file = np.array(ann_file)
        labels = ann_file[:, 0].astype(int)
        boxes = ann_file[:, 1:]
        return boxes.tolist(), labels.tolist()

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = read_image(os.path.join(self.image_path, self.image_files[image_id]))

        if image.shape[0] > 3:
            image = image[:3, :, :]
        height, width = image.size()[1:3]
        image = resize(image)

        _boxes = []
        for box in boxes:
            x, y, w, h = scale_bbox(box, width, height)
            x1, y1, x2, y2 = yolobbox2bbox(bbox=[x, y, w, h])
            _boxes.append([x1, y1, x2, y2])
        boxes = np.array(_boxes)

        labels = torch.tensor([int(l) for l in labels])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {"boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY",
                                                    canvas_size=F.get_size(image)),
                  "labels": labels,
                  "area": area,
                  "iscrowd": iscrowd}
        return image, target

    def __len__(self):
        return len(self.ids)
