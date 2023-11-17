import math
import sys

from tqdm import tqdm

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


args = parse_args()
dataset = YOLODataset(image_path=args.image_path,
                      annotation_path=args.annotation_path,
                      label_file=args.label_file,
                      shuffle=args.shuffle)

dataset_val = YOLODataset(image_path=args.image_path,
                      annotation_path=args.annotation_path,
                      label_file=args.label_file,
                      shuffle=args.shuffle)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=1,
    collate_fn=collate_fn,
    drop_last=True
)
data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=2,
    shuffle=False,
    num_workers=1,
    collate_fn=collate_fn,
    drop_last=True
)

model = get_model()




# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
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
num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    model.train()
    loss_value=0
    for idx, (images, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = list(image for image in images)
        output = model(images, targets)  # Returns losses and detections
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        # print(loss_value)
        if lr_scheduler is not None:
            lr_scheduler.step()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_value,
    }, 'checkpoints')


print("That's it!")
def main():
    pass


if __name__ == "__main__":
    main()
