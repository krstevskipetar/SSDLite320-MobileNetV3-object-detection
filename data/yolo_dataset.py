import os
import random
from os.path import exists, join
from typing import Sequence

import numpy as np
import torch.utils.data
import torchvision.transforms
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F

random.seed(42)
np.random.seed(42)


class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, image_path, annotation_path, label_file, transform=None, target_transforms=None, shuffle=False,
                 resize_to=(320, 320)):

        self.image_path = image_path
        self.annotation_path = annotation_path
        self.transform = transform
        self.target_transforms = target_transforms

        self.image_files = os.listdir(image_path)
        if shuffle:
            random.shuffle(self.image_files)

        self.annotation_files = [''.join(img.split('.')[:-1]) + '.txt' for img in self.image_files]
        self._validate_input()
        self.ids = [i for i in range(len(self.image_files))]
        self.resize_to = resize_to
        self.resized_h, self.resized_w = resize_to
        self.resize = torchvision.transforms.Resize(resize_to, antialias=True)

        with open(label_file, 'r') as f:
            classes = f.readlines()
            classes = [c.strip() for c in classes]
            self.class_names = classes

        self.class_dict = {class_name: i for i, class_name in
                           enumerate(self.class_names)}

    @staticmethod
    def yolobbox2bbox(bbox: Sequence[int]):
        x, y, w, h = bbox
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def scale_bbox(bbox: Sequence[int], image_width: int, image_height: int):
        x, y, w, h = bbox
        x *= image_width
        w *= image_width
        y *= image_height
        h *= image_height
        return x, y, w, h

    @staticmethod
    def resize_bbox(bbox: Sequence[int], x_factor, y_factor):
        x1, y1, x2, y2 = bbox
        x1 *= x_factor
        x2 *= x_factor
        y1 *= y_factor
        y2 *= y_factor
        return x1, y1, x2, y2

    def _validate_input(self):
        valid_indices = []
        removed_count = 0

        for idx, (image_file, annotation_file) in enumerate(zip(self.image_files, self.annotation_files)):
            image_path = os.path.join(self.image_path, image_file)
            annotation_path = os.path.join(self.annotation_path, annotation_file)

            if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                removed_count += 1
                continue

            with open(annotation_path, 'r') as f:
                ann_file = f.readlines()
                if len(ann_file) == 0:
                    removed_count += 1
                    continue

            try:
                image = read_image(image_path)
                boxes, labels = self._get_annotation(idx)
                if np.shape(boxes)[1] > 4:
                    removed_count += 1
                    continue
            except (IndexError, ValueError, RuntimeError):
                removed_count += 1
                continue

            valid_indices.append(idx)

        self.image_files = [self.image_files[i] for i in valid_indices]
        self.annotation_files = [self.annotation_files[i] for i in valid_indices]

        print(f"Removed {removed_count} invalid samples")

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
        image = image.float()
        if image.shape[0] > 3:
            image = image[:3, :, :]
        height, width = image.size()[1:3]
        image = self.resize(image)
        image = image.float()
        image /= 255

        _boxes = []
        for box in boxes:
            x, y, w, h = self.scale_bbox(box, width, height)
            x1, y1, x2, y2 = self.yolobbox2bbox(bbox=[x, y, w, h])
            x1, y1, x2, y2 = self.resize_bbox([x1, y1, x2, y2], self.resized_w / width, self.resized_h / height)
            _boxes.append([x1, y1, x2, y2])
        boxes = np.array(_boxes)

        labels = torch.tensor([int(l) for l in labels])
        target = {"boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY",
                                                    canvas_size=F.get_size(image)),
                  "labels": labels,
                  "image_id": image_id}
        return image, target

    def __len__(self):
        return len(self.ids)
