import torch
import torchvision.ops as ops
from tqdm import tqdm
import time
from postprocessing import apply_nms


def calculate_precision_recall(predictions, targets, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    true_positives = 0
    false_positives = 0
    total_gt_boxes = len(targets['boxes'])

    if total_gt_boxes == 0:
        return 0, 0

    for iou_threshold in iou_thresholds:
        keep = apply_nms(predictions['boxes'], predictions['scores'], threshold=0.2, score_threshold=0.05)
        kept_predictions = {'boxes': predictions['boxes'][keep],
                            'labels': predictions['labels'][keep]}
        for pred_boxes, pred_labels in zip(kept_predictions['boxes'], kept_predictions['labels']):
            max_iou = 0
            matched_gt_index = -1

            for i, (gt_box, gt_label) in enumerate(zip(targets['boxes'], targets['labels'])):
                iou = calculate_iou(pred_boxes, gt_box)
                if iou > max_iou and gt_label == pred_labels:
                    max_iou = iou
                    matched_gt_index = i

            if max_iou >= iou_threshold:
                true_positives += 1
                targets['boxes'] = torch.cat((targets['boxes'][:matched_gt_index],
                                              targets['boxes'][matched_gt_index + 1:]), dim=0)
            else:
                false_positives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_gt_boxes

    return precision, recall


def calculate_iou(bbox1, bbox2):
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2

    intersection_area = max(0, min(x_max1, x_max2) - max(x_min1, x_min2)) * max(0, min(y_max1, y_max2) - max(y_min1,
                                                                                                             y_min2))
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def calculate_ap_ar_map(predictions, ground_truth, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    total_precision = 0
    total_recall = 0
    total_map = 0

    for preds, gt in zip(predictions, ground_truth):
        precision, recall = calculate_precision_recall(preds, gt, iou_thresholds)
        total_precision += precision
        total_recall += recall
        total_map += calculate_average_precision(preds, gt, iou_thresholds)

    num_samples = len(predictions)
    ap = total_precision / num_samples if num_samples > 0 else 0
    ar = total_recall / num_samples if num_samples > 0 else 0
    map = total_map / num_samples if num_samples > 0 else 0

    return ap, ar, map


def calculate_average_precision(predictions, targets, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    true_positives = 0
    false_positives = 0
    total_gt_boxes = len(targets)
    precision_values = []

    if total_gt_boxes == 0:
        return 0

    for iou_threshold in iou_thresholds:
        keep = apply_nms(predictions['boxes'], predictions['scores'], threshold=0.2, score_threshold=0.05)
        kept_predictions = {'boxes': predictions['boxes'][keep],
                            'labels': predictions['labels'][keep]}
        for i, (pred_boxes, pred_labels) in enumerate(zip(kept_predictions['boxes'], kept_predictions['labels'])):
            max_iou = 0
            matched_gt_index = -1

            for j, (gt_box, gt_label) in enumerate(zip(targets['boxes'], targets['labels'])):
                iou = calculate_iou(pred_boxes, gt_box)
                if iou > max_iou and gt_label == pred_labels:
                    max_iou = iou
                    matched_gt_index = j

            if max_iou >= iou_threshold:
                true_positives += 1
                targets['boxes'] = torch.cat((targets['boxes'][:matched_gt_index],
                                              targets['boxes'][matched_gt_index + 1:]), dim=0)
            else:
                false_positives += 1

            precision = true_positives / (true_positives + false_positives)
            precision_values.append(precision)

    average_precision = sum(precision_values) / len(precision_values) if precision_values else 0
    return average_precision
