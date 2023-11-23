# SSDLite320 MobileNetV3 object detection

This repository contains code for training and evaluating an SSDLite-320 object detector, utilizing MobileNetV3 as a
feature extractor.

## Licensing Information

### PyTorch Code (vision directory)

The code in the `vision` directory, which originates from PyTorch, is licensed under
the [BSD-3-Clause License](vision/LICENSE_PyTorch). This code has been modified.

### Other code (root directory)

The code in the root directory, written by Petar Krstevski, is licensed under the [MIT License](LICENSE).

Please refer to the individual source files for specific licensing information.

## Usage

To start training, run the following command:

```
python train.py --image_path path/to/train/images --annotation_path path/to/train/annotations \
 --image_path_val path/to/val/images  --annotation_path_val path/to/val/annotations \
 --label_file path/to/label_file.txt --shuffle False --device cuda:0
 ```

The dataset annotations should be in YOLO
format (https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format),
i.e. for each image example.jpg there should be a corresponding annotation file example.txt containing bounding box
annotations in
a normalized XYWH format. The annotation file should contain one row per object in class x_center y_center width height
format.

The YOLODataset class then reformats the XYWH input annotations into an unnormalized XYXY format for compatibility with
torch models, which use the COCO format.

