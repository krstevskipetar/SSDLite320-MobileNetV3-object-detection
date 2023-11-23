from matplotlib import pyplot as plt
from torch.utils.data import Subset
import pickle
from vision.references.detection.coco_utils import get_coco_api_from_dataset
from vision.references.detection.utils import collate_fn
from yolo_dataset import YOLODataset
import torch
import argparse
from model import get_model
import numpy as np
import random
from vision.references.detection.engine import evaluate, train_epoch

plt.ion()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='/home/petar/waste_dataset_v2/train/images')
    parser.add_argument('--annotation_path', default='/home/petar/waste_dataset_v2/train/labels')
    parser.add_argument('--image_path_val', default='/home/petar/waste_dataset_v2/val/images')
    parser.add_argument('--annotation_path_val', default='/home/petar/waste_dataset_v2/val/labels')
    parser.add_argument('--label_file', default='/home/petar/waste_dataset_v2/label_map.txt')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--device', default='cpu')
    argz = parser.parse_args()

    return argz


args = parse_args()

with open(args.label_file, 'r') as f:
    classes = f.readlines()
    classes = [c.strip() for c in classes]
    class_names = classes

class_dict = {class_name: i for i, class_name in enumerate(class_names)}

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
    batch_size=2,
    shuffle=True,
    num_workers=1,
    collate_fn=collate_fn,
    drop_last=True
)
data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    # Subset(dataset, random.sample([i for i in range(len(dataset_val))], 100)),
    batch_size=2,
    shuffle=False,
    num_workers=1,
    collate_fn=collate_fn,
    drop_last=True
)

model = get_model(num_classes=4, trainable_backbone_layers=6, weights=None)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=10,
                                                          eta_min=0)

num_epochs = 15
print_freq = 10
evaluate_every = 5

coco_subsets = {'val': get_coco_api_from_dataset(data_loader_val.dataset),
                'train:': get_coco_api_from_dataset(data_loader.dataset)}

all_losses = {}
evaluators, loggers = [], []
best_loss = np.inf
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    all_losses, epoch_loss = train_epoch(model=model,
                                         optimizer=optimizer,
                                         lr_scheduler=lr_scheduler,
                                         data_loader=data_loader,
                                         print_freq=print_freq,
                                         epoch_number=epoch,
                                         all_losses=all_losses)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, f'checkpoints/epoch_{epoch}.pth')

    if epoch != 0 and epoch % evaluate_every == 0:
        print("Evaluation on val set")
        evltr, lgr = evaluate(model, data_loader_val, torch.device(args.device), coco_subsets['val'])
        evaluators.append(evltr)
        loggers.append(lgr)

with open('evltr.pkl', 'wb') as f:
    pickle.dump(evaluators, f)
with open('lgr.pkl', 'wb') as f:
    pickle.dump(loggers, f)


def main():
    pass


if __name__ == "__main__":
    main()
