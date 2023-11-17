import pdb
import random

import numpy as np
import torchvision.transforms
import os

import torch.utils.data
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


def yolobbox2bbox(x, y, w, h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2


resize = torchvision.transforms.Resize(640, antialias=True)


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

        with open(label_file, 'r') as f:
            classes = f.readlines()
            classes = [c.strip() for c in classes]
            self.class_names = classes

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

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
        image = image.float()
        size = image.size()[1:3]
        image = resize(image)
        image = tv_tensors.Image(image)

        x_scale = size[0] / image.size()[1] * 640
        y_scale = size[1] / image.size()[2] * 640
        # Update bounding box coordinates
        boxes = [bb * torch.tensor([x_scale, y_scale, x_scale, y_scale]) for bb in torch.tensor(boxes)]
        boxes_ = []
        for box in boxes:
            boxes_.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
        boxes = np.array(boxes_)

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
