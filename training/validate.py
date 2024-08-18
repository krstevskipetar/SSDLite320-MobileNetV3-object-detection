import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset

from core.engine import validate
from core.model import get_model
from core.plotting import plot_predictions_in_grid
from core.wandb_logging import log_to_wandb
from data.load_data import load_class_names
from data.yolo_dataset import YOLODataset
from vision.references.detection.utils import collate_fn

plt.ion()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_val', default='/home/petar/waste_dataset_v2_refactored/val/images')
    parser.add_argument('--annotation_path_val', default='/home/petar/waste_dataset_v2_refactored/val/labels')
    parser.add_argument('--label_file', default='/home/petar/waste_dataset_v2_refactored/label_map.txt')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default="checkpoints/checkpoints/epoch_99.pth")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--wandb_logging', action='store_true', default=False)
    parser.add_argument('--wandb_project_name', default=None)
    argz = parser.parse_args()

    return argz


def create_unique_folder(base_path, prefix='_idx'):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    folder_name = f"{prefix}{timestamp}"
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path


def infer_and_plot_batch_predictions(model, data_loader_val, class_names, n_plots=5):
    figs = []
    run = create_unique_folder("../batch_predictions")
    for idx, (images, targets) in enumerate(data_loader_val):
        predictions = model(images)
        fig = plot_predictions_in_grid(images, predictions, targets, class_names, show_plot=False)
        figs.append(fig)
        fig.savefig(f"{run}/batch_predictions_{idx}.png")
        if idx != 0 and (idx + 1) % n_plots == 0:
            break
    return figs


def run_validation(checkpoint, image_path_val, annotation_path_val, label_file, device='cpu', num_classes=5,
                   shuffle=True, wandb_logging=False, wandb_project_name=None, iou_thresholds=None):
    class_names = load_class_names(label_file)

    dataset_val = YOLODataset(image_path=image_path_val,
                              annotation_path=annotation_path_val,
                              label_file=label_file,
                              shuffle=shuffle)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        drop_last=True
    )

    device = torch.device(device)
    model = get_model(num_classes=num_classes)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state_dict'])
    model.eval()

    iou_thresholds = np.arange(0.05, 0.55, 0.05) if iou_thresholds is None else iou_thresholds
    mean_ap, class_precisions = validate(model=model,
                                         data_loader=data_loader_val,
                                         class_names=class_names,
                                         device=device,
                                         iou_thresholds=iou_thresholds)
    print("For iou thresholds", iou_thresholds)
    print(f"Mean Average Precision: {mean_ap}")

    print("Class precisions: ")
    for key, value in class_precisions.items():
        print(f"\t-{key}: {value}")

    if wandb_logging:
        config = {
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
            "checkpoint": checkpoint
        }
        metrics = {'mAP@5:50': mean_ap}
        metrics.update(
            {str(class_name) + '_precision': class_precision for class_name, class_precision in
             class_precisions.items()})
        figures = infer_and_plot_batch_predictions(model, data_loader_val,
                                                   {i + 1: c for (i, c) in enumerate(class_names)},
                                                   5)
        if wandb_project_name:
            log_to_wandb(wandb_project_name, config, metrics, figures, checkpoint)
    return mean_ap, class_precisions


def main():
    args = parse_args()
    _, _ = run_validation(args.checkpoint, args.image_path_val, args.annotation_path_val, args.label_file,
                          args.device,
                          args.num_classes, args.shuffle, args.wandb_logging,
                          args.wandb_project_name)


if __name__ == "__main__":
    main()
