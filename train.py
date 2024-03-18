import os
from datetime import datetime
from os.path import join

from matplotlib import pyplot as plt

from core.engine import validate, train_epoch
from data.load_data import load_train_and_val_datasets, create_train_and_val_dataloaders, load_class_names
from core.wandb_logging import log_to_wandb
from validate import infer_and_plot_batch_predictions
import torch
import argparse
from core.model import get_model
import numpy as np
import random

# from vision.references.detection.engine import evaluate, train_epoch

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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--evaluate_every', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--trainable_backbone_layers', type=int, default=6)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--wandb_logging', default=True)
    parser.add_argument('--wandb_project_name', default='SSDLite320-MobileNetV3 Waste Classification')
    return parser.parse_args()


def create_unique_folder(base_path, prefix='_idx'):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    folder_name = f"{prefix}{timestamp}"
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path


def run_training(data_loader_train, data_loader_val, model, optimizer, lr_scheduler, class_names,
                 num_epochs=100, device='cpu', print_freq=100, evaluate_every=5):
    checkpoint_path = create_unique_folder('checkpoints', 'run')
    all_losses = {}
    metrics = {key: [] for key in ['mAP@50', 'mAR@50']}
    last_checkpoint = ''
    for epoch in range(num_epochs):
        epoch = epoch + 1
        # train for one epoch, printing every 10 iterations
        all_losses, epoch_loss = train_epoch(model=model,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             data_loader=data_loader_train,
                                             device=device,
                                             print_freq=print_freq,
                                             epoch_number=epoch,
                                             all_losses=all_losses)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f'checkpoints/epoch_{epoch}.pth')
        if epoch != 0 and epoch % evaluate_every == 0:
            print("Evaluation on val set")
            mean_ap, mean_ar, class_precisions, class_recalls = validate(model, data_loader_val, class_names,
                                                                         torch.device(device), [0.5])
            metrics['mAP@50'].append(mean_ap)
            metrics['mAR@50'].append(mean_ar)

            print("For iou thresholds", 0.5)
            print(f"Mean Average Precision: {mean_ap}\nMean Average Recall: {mean_ar}")

            print("Class precisions: ")
            for key, value in class_precisions.items():
                print(f"\t-{key}: {value}")

            print("Class recalls: ")
            for key, value in class_recalls.items():
                print(f"\t-{key}: {value}")

        last_checkpoint = join(checkpoint_path, f'/epoch_{epoch}.pth')

    figures = infer_and_plot_batch_predictions(model, data_loader_val, class_names, 5)
    return metrics, figures, last_checkpoint


def main():
    args = parse_args()
    class_names = load_class_names(args.label_file)

    dataset_train, dataset_val = load_train_and_val_datasets(image_path_train=args.image_path,
                                                             annotation_path_train=args.annotation_path,
                                                             image_path_val=args.image_path_val,
                                                             annotation_path_val=args.annotation_path_val,
                                                             label_file=args.label_file,
                                                             shuffle_train=args.shuffle,
                                                             shuffle_val=args.shuffle)

    data_loader_train, data_loader_val = create_train_and_val_dataloaders(dataset_train=dataset_train,
                                                                          dataset_val=dataset_val,
                                                                          batch_size_train=args.batch_size,
                                                                          batch_size_val=args.batch_size,
                                                                          shuffle_train=True,
                                                                          shuffle_val=False,
                                                                          num_workers=1)

    model = get_model(num_classes=args.num_classes,
                      trainable_backbone_layers=args.trainable_backbone_layers,
                      weights=args.weights)

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

    metrics, figures, last_checkpoint = run_training(data_loader_train=data_loader_train,
                                                     data_loader_val=data_loader_val,
                                                     model=model,
                                                     optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler,
                                                     class_names=class_names,
                                                     device=args.device,
                                                     print_freq=args.print_freq,
                                                     evaluate_every=args.evaluate_every)

    if args.wandb_logging:
        config = {
            "run_type": "training",
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
            "epochs": args.num_epochs,
            "checkpoint": last_checkpoint
        }
        log_to_wandb(args.wandb_project_name, config, metrics, figures, last_checkpoint)


if __name__ == "__main__":
    main()
