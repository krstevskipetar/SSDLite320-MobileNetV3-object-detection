import math
import sys

import numpy as np
import torch
from tqdm import tqdm

from vision.references.detection import utils
from metrics import calculate_ap_ar_map
import time


def validate(model, data_loader, class_names, device, iou_thresholds=None):
    model.eval()
    model = model.to(device)
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.1, 1, 0.1)
    all_ap, all_ar, all_map = [], [], []
    all_class_precisions, all_class_recalls = ({class_name: [] for class_name in class_names},
                                               {class_name: [] for class_name in class_names})
    start_time = time.time()
    for idx, (images, targets) in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True):
        images = [image.to(device) for image in images]
        output = model(images)

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        mean_ap, mean_ar, class_precisions, class_recalls = calculate_ap_ar_map(output, targets, class_names,
                                                                                iou_thresholds)

        for key in class_names:
            all_class_precisions[key].append(class_precisions[key])
            all_class_recalls[key].append(class_recalls[key])

        all_ap.append(mean_ap)
        all_ar.append(mean_ar)
    end_time = time.time()
    print("Inference took ", end_time - start_time, " seconds")
    mean_ap, mean_ar = np.mean(all_ap), np.mean(all_ar)
    for key in class_names:
        all_class_precisions[key] = np.mean(all_class_precisions[key])
        all_class_recalls[key] = np.mean(all_class_recalls[key])

    return mean_ap, mean_ar, all_class_precisions, all_class_recalls


def train_epoch(model, optimizer, lr_scheduler, data_loader, device, print_freq=100, epoch_number=0, all_losses=None):
    model.train()
    local_losses = []
    device = torch.device(device)
    for idx, (images, targets) in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True):

        optimizer.zero_grad()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # Returns losses and detections
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        local_losses.append(loss_value)
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        if idx % print_freq == 0 and idx != 0:
            print("Loss: ", loss_value)

        losses.backward()
        optimizer.step()
        # print(loss_value)
        if lr_scheduler is not None:
            lr_scheduler.step()
    all_losses.update({epoch_number: local_losses})

    return all_losses, np.mean(local_losses)
