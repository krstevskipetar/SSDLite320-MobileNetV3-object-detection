import torchvision
from torch import Tensor


def apply_nms(boxes: Tensor, scores: Tensor, threshold: float = 0.5, score_threshold: float = 0.2):
    index = scores > score_threshold
    boxes = boxes[index]
    scores = scores[index]
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=threshold)
    return keep
