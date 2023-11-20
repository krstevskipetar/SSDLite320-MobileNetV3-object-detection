import math
import sys

from tqdm import tqdm
import torchvision

print(torchvision.__version__)
from utils import collate_fn, reduce_dict
from yolo_dataset import YOLODataset
import torch
import argparse
from model import get_model
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='/home/petar/waste_dataset_v2/train/images')
    parser.add_argument('--annotation_path', default='/home/petar/waste_dataset_v2/train/labels')
    parser.add_argument('--image_path_val', default='/home/petar/waste_dataset_v2/val/images')
    parser.add_argument('--annotation_path_val', default='/home/petar/waste_dataset_v2/val/labels')
    parser.add_argument('--label_file', default='/home/petar/waste_dataset_v2/label_map.txt')
    parser.add_argument('--shuffle', action='store_true', default=False)
    argz = parser.parse_args()

    return argz


import random
from torch.utils.data import Subset

args = parse_args()
dataset = YOLODataset(image_path=args.image_path,
                      annotation_path=args.annotation_path,
                      label_file=args.label_file,
                      shuffle=args.shuffle)

dataset_val = YOLODataset(image_path=args.image_path_val,
                          annotation_path=args.annotation_path_val,
                          label_file=args.label_file,
                          shuffle=args.shuffle)

data_loader = torch.utils.data.DataLoader(
    dataset,
    # Subset(dataset, random.sample([i for i in range(len(dataset))], 100)),
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    drop_last=True
)
data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    # Subset(dataset, random.sample([i for i in range(len(dataset_val))], 100)),

    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    drop_last=True
)

model = get_model(trainable_backbone_layers=6)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.0001,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it for 5 epochs
num_epochs = 12
print_freq = 10

from engine import evaluate


def train_epoch():
    model.train()
    loss_value = 0
    for idx, (images, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = list(image for image in images)
        loss_dict = model(images, targets)  # Returns losses and detections
        classification_loss = loss_dict['classification']
        regression_loss = loss_dict['bbox_regression']
        losses = sum(loss for loss in loss_dict.values())
        loss = torch.sum(torch.stack((classification_loss, regression_loss)))
        loss_value = loss.item()
        classification_loss.backward(retain_graph=True)
        regression_loss.backward()
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_value)
            sys.exit(1)
        if idx % print_freq == 0:
            print("Loss: ", loss_value)
        optimizer.zero_grad()
        # losses.backward()
        optimizer.step()
        # print(loss_value)
        if lr_scheduler is not None:
            lr_scheduler.step()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_value,
    }, f'checkpoints/epoch_{epoch}.pth')


def validate():
    evaluate(model, data_loader_val, torch.device('cpu'))


for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_epoch()
    print("Evaluation on train set")
    evaluate(model, data_loader, torch.device('cpu'))
    print("Evaluation on val set")

    evaluate(model, data_loader_val, torch.device('cpu'))

print("That's it!")


def main():
    pass


if __name__ == "__main__":
    main()
