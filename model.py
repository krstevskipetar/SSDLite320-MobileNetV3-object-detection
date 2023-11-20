from functools import partial

import torchvision
from torch import nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead, SSDLite320_MobileNet_V3_Large_Weights
import _utils as det_utils

# todo: Change model head and load pretrained weights for fine-tuning
def get_model(num_classes=4, trainable_backbone_layers=4):
    model = ssdlite320_mobilenet_v3_large(num_classes=num_classes,
                                          # weights='DEFAULT',
                                          trainable_backbone_layers=trainable_backbone_layers)

    # weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    # weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    # norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    # backbone = mobilenet_v3_large(
    #     weights=weights_backbone, progress=False, norm_layer=norm_layer, reduced_tail=False)
    #
    # anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    # size = (320, 320)
    #
    # out_channels = det_utils.retrieve_out_channels(backbone, size)
    # num_anchors = anchor_generator.num_anchors_per_location()
    # model.head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)
    return model
