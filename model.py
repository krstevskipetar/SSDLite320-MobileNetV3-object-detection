from functools import partial

import torchvision
from torch import nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead, SSDLite320_MobileNet_V3_Large_Weights
import _utils as det_utils


# todo: Change model head and load pretrained weights for fine-tuning
def get_model(num_classes=4, trainable_backbone_layers=6, weights=None):
    model = ssdlite320_mobilenet_v3_large(weights=weights, num_classes=num_classes, trainable_backbone_layers=None)

    if weights is None:
        return model
    else:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
        num_anchors = anchor_generator.num_anchors_per_location()
        model.head = SSDLiteHead([672], num_anchors, num_classes, norm_layer)
    return model
