from collections import Counter

import torch

from core.postprocessing import apply_nms


def calculate_precision_recall(class_id, class_predictions, class_ground_truth, iou_threshold):
    detections = []
    ground_truths = []
    for i, prediction in enumerate(class_predictions):
        for j, box in enumerate(prediction['boxes']):
            x1, y1, x2, y2 = box
            prob = prediction['scores'][j]
            detections.append([i, class_id, prob, x1, y1, x2, y2])
    for i, gt in enumerate(class_ground_truth):
        for j, box in enumerate(gt['boxes']):
            x1, y1, x2, y2 = box
            ground_truths.append([i, class_id, 1, x1, y1, x2, y2])

    amount_bboxes = Counter([gt[0] for gt in ground_truths])
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)
    detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros(len(detections))
    FP = torch.zeros(len(detections))
    total_true_boxes = len(ground_truths)
    for detection_idx, detection in enumerate(detections):
        ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
        best_iou = 0
        best_gt_idx = None
        for idx, gt in enumerate(ground_truth_img):
            iou = calculate_iou(detection[3:], gt[3:])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou > iou_threshold:
            if best_gt_idx is not None:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1  # bbox covered
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        else:
            FP[detection_idx] = 1
    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_boxes + 1e-6)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    return precisions, recalls


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


def calculate_mean_ap(predictions, ground_truth, class_names, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = torch.linspace(0.5, 0.95, 10)  # Standard COCO IoU thresholds
    kept_predictions = []
    for pred in predictions:
        keep = apply_nms(pred['boxes'], pred['scores'], threshold=0.2, score_threshold=0.05)
        kept_predictions.append({'boxes': pred['boxes'][keep],
                                 'labels': pred['labels'][keep],
                                 'scores': pred['scores'][keep]})
    predictions = kept_predictions

    num_classes = len(class_names)
    class_name_dict = {i + 1: class_name for (i, class_name) in enumerate(class_names)}
    class_average_precisions = {class_name_dict[i]: [] for i in range(1, num_classes + 1, 1)}
    mean_average_precisions = []
    ps = {class_name_dict[i]: [] for i in range(1, num_classes + 1, 1)}
    rs = {class_name_dict[i]: [] for i in range(1, num_classes + 1, 1)}
    for iou_threshold in iou_thresholds:
        average_precisions = []
        for class_id in range(1, num_classes + 1, 1):
            class_predictions = filter_values_by_class(values=predictions, class_id=class_id)
            class_ground_truth = filter_values_by_class(values=ground_truth, class_id=class_id)
            precisions, recalls = calculate_precision_recall(class_id, class_predictions, class_ground_truth,
                                                             iou_threshold)
            ap = torch.trapz(precisions, recalls)
            average_precisions.append(ap)
            class_average_precisions[class_name_dict[class_id]].append(ap)
            ps[class_name_dict[class_id]].append(precisions[-1])
            rs[class_name_dict[class_id]].append(recalls[-1])
        mean_average_precisions.append(sum(average_precisions) / len(average_precisions))

    for key in class_average_precisions.keys():
        class_average_precisions[key] = sum(class_average_precisions[key]) / len(class_average_precisions[key])
    for key in ps.keys():
        ps[key] = sum(ps[key]) / len(ps[key])
    for key in rs.keys():
        rs[key] = sum(rs[key]) / len(rs[key])

    return sum(mean_average_precisions) / len(mean_average_precisions), class_average_precisions, ps, rs
