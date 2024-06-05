import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from tqdm import tqdm

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
    parser.add_argument('--checkpoints_path', default="checkpoints")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--wandb_logging', action='store_true', default=False)
    parser.add_argument('--wandb_project_name')
    parser.add_argument('--output_path', default='./')
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
    run = create_unique_folder("batch_predictions")
    for idx, (images, targets) in enumerate(data_loader_val):
        predictions = model(images)
        fig = plot_predictions_in_grid(images, predictions, targets, class_names, show_plot=False)
        figs.append(fig)
        fig.savefig(f"{run}/batch_predictions_{idx}.png")
        if idx != 0 and (idx + 1) % n_plots == 0:
            break
    return figs


def main():
    args = parse_args()

    class_names = load_class_names(args.label_file)
    all_class_precisions = {cn: {} for cn in class_names}
    all_class_recalls = {cn: {} for cn in class_names}
    mean_aps, mean_ars = {}, {}
    dataset_val = YOLODataset(image_path=args.image_path_val,
                              annotation_path=args.annotation_path_val,
                              label_file=args.label_file,
                              shuffle=args.shuffle)

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
    model = get_model(num_classes=args.num_classes)

    checkpoints = os.listdir(args.checkpoints_path)
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
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
        "id": None
    }
    for epoch_idx, checkpoint in enumerate(checkpoints):
        print(checkpoint)
        model.load_state_dict(
            torch.load(os.path.join(args.checkpoints_path, checkpoint), map_location=device)['model_state_dict'])
        model.eval()
        iou_thresholds = [0.5]
        mean_ap, mean_ar, class_precisions, class_recalls = validate(model=model,
                                                                     data_loader=data_loader_val,
                                                                     class_names=class_names,
                                                                     device=device,
                                                                     iou_thresholds=iou_thresholds)
        mean_aps.update({epoch_idx: mean_ap})
        mean_ars.update({epoch_idx: mean_ar})

        print("For iou thresholds", iou_thresholds)
        print(f"Mean Average Precision: {mean_ap}\nMean Average Recall: {mean_ar}")

        print("Class precisions: ")
        for key, value in class_precisions.items():
            all_class_precisions[key].update({epoch_idx: value})
            print(f"\t-{key}: {value}")

        print("Class recalls: ")
        for key, value in class_recalls.items():
            all_class_recalls[key].update({epoch_idx: value})
            print(f"\t-{key}: {value}")
        if args.wandb_logging:
            metrics_to_log = {"mAP@50": mean_ap, "mAR@50": mean_ar}
            for key, value in class_precisions.items():
                metrics_to_log.update({key+"_precision": value})
            for key, value in class_recalls.items():
                metrics_to_log.update({key+"_recall": value})

            log_to_wandb(args.wandb_project_name, config, metrics_to_log, [],
                         os.path.join(args.checkpoints_path, checkpoint), step=epoch_idx, plot=False)

    metrics = {'mAP@50': mean_aps,
               'mAR@50': mean_ars}
    metrics.update(
        {str(class_name) + '_precision': class_precision for class_name, class_precision in
         all_class_precisions.items()})
    metrics.update(
        {str(class_name) + '_recall': class_recall for class_name, class_recall in all_class_recalls.items()})

    with open(os.path.join(args.output_path, "metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file)
    with open(os.path.join(args.output_path, "config.json"), 'w') as config_file:
        json.dump(config, config_file)


if __name__ == "__main__":
    main()
