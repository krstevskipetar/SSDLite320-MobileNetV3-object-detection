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
from metrics import calculate_ap_ar_map
from engine import validate
from plotting import plot_predictions_in_grid

plt.ion()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_val', default='/home/petar/waste_dataset_v2/val/images')
    parser.add_argument('--annotation_path_val', default='/home/petar/waste_dataset_v2/val/labels')
    parser.add_argument('--label_file', default='/home/petar/waste_dataset_v2/label_map.txt')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--wandb_logging', action='store_true', default=False)
    argz = parser.parse_args()

    return argz


args = parse_args()

with open(args.label_file, 'r') as f:
    classes = f.readlines()
    classes = [c.strip() for c in classes]
    class_names = classes

class_dict = {class_name: i for i, class_name in enumerate(class_names)}

dataset_val = YOLODataset(image_path=args.image_path_val,
                          annotation_path=args.annotation_path_val,
                          label_file=args.label_file,
                          shuffle=args.shuffle,
                          device=args.device)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    # Subset(dataset, random.sample([i for i in range(len(dataset_val))], 100)),
    batch_size=9,
    shuffle=False,
    num_workers=1,
    collate_fn=collate_fn,
    drop_last=True
)
device = torch.device(args.device)
model = get_model(num_classes=4, trainable_backbone_layers=6, weights=None)
if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
model.eval()
ap, ar, mean_ap = validate(model=model,
                           data_loader=data_loader_val,
                           device=device,
                           iou_thresholds=[0.5])

n_plots = 5
figs = []
for idx, (images, targets) in enumerate(data_loader_val):
    predictions = model(images)
    fig = plot_predictions_in_grid(images, predictions, targets, class_names, show_plot=False)
    figs.append(fig)
    fig.savefig(f"batch_predictions_{idx}.png")
    if idx != 0 and (idx + 1) % n_plots == 0:
        break
if args.wandb_logging:
    import wandb

    wandb.init(
        # set the wandb project where this run will be logged
        project="SSDLite320-MobileNetV3 Waste Classification",

        # track hyperparameters and run metadata
        config={
            "run_type": "validation",
            "optimizer": "SGD",
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "lr_scheduler": "CosineAnnealingLR",
            "T_max": 10,
            "eta_min": 0,
            "learning_rate": 0.001,
            "detector": "SSDLite",
            "backbone": "MobileNetV3",
            "dataset": "waste-dataset-v2",
            "epochs": 100,
            "checkpoint": "epoch_99.pth"
        }
    )
    wandb.log({"average_precision": ap,
               "average_recall": ar,
               "mAP@50": mean_ap})
    for idx, f in enumerate(figs):
        wandb.log({f"batch_predictions_{idx}": f})
    wandb.log_artifact("checkpoints/checkpoints/epoch_99.pth")
