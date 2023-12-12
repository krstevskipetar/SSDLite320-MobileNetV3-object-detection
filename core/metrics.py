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
    precision = true_positives / (true_positives + false_positives + 1e-12)
    recall = true_positives / (total_gt_boxes + 1e-12)

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


def filter_values_by_class(values: list, class_id: int):
    filtered_values = []
    for val in values:
        index = val['labels'] == class_id
        if not index.any():
            continue
        filtered_val = {}
        for key in val.keys():
            if key == 'image_id':
                filtered_val.update({key: val[key]})
            else:
                filtered_val.update({key: val[key][index]})
        filtered_values.append(filtered_val)
    return filtered_values


def calculate_ap_ar_map(predictions, ground_truth, class_names, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    kept_predictions = []
    for pred in predictions:
        keep = apply_nms(pred['boxes'], pred['scores'], threshold=0.2, score_threshold=0.05)
        kept_predictions.append({'boxes': pred['boxes'][keep],
                                 'labels': pred['labels'][keep],
                                 'scores': pred['scores'][keep]})
    predictions = kept_predictions
    total_precision = 0
    total_recall = 0
    num_classes = len(class_names)
    class_name_dict = {i: class_name for (i, class_name) in enumerate(class_names)}
    class_precisions, class_recalls = {}, {}
    for class_id in range(1, num_classes, 1):
        class_predictions = filter_values_by_class(values=predictions, class_id=class_id)
        class_ground_truth = filter_values_by_class(values=ground_truth, class_id=class_id)
        total_precision_class = 0
        total_recall_class = 0
        for preds, gt in zip(class_predictions, class_ground_truth):
            precision, recall = calculate_precision_recall(preds, gt, iou_thresholds)
            total_precision_class += precision
            total_recall_class += recall

        num_samples_class = len(class_predictions)
        ap_class = total_precision_class / num_samples_class if num_samples_class > 0 else 0
        ar_class = total_recall_class / num_samples_class if num_samples_class > 0 else 0

        class_precisions.update({class_name_dict[class_id]: ap_class})
        class_recalls.update({class_name_dict[class_id]: ar_class})

        total_precision += ap_class
        total_recall += ar_class

    mean_ap = total_precision / num_classes if num_classes > 0 else 0
    mean_ar = total_recall / num_classes if num_classes > 0 else 0

    return mean_ap, mean_ar, class_precisions, class_recalls
