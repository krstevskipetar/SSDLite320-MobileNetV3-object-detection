import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


def get_model(num_classes=4, trainable_backbone_layers=4):
    model = ssdlite320_mobilenet_v3_large(num_classes=num_classes,
                                          trainable_backbone_layers=trainable_backbone_layers)
    return model
