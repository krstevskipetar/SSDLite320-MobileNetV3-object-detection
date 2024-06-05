#!/bin/bash

python validate_all_epochs.py \
    --image_path_val '/home/petar/waste_dataset_v2_refactored/val/images' \
    --annotation_path_val '/home/petar/waste_dataset_v2_refactored/val/labels' \
    --label_file '/home/petar/waste_dataset_v2_refactored/label_map.txt' \
    --shuffle \
    --device 'cpu' \
    --checkpoints_path 'checkpoints' \
    --num_classes 5 \
    --wandb_logging \
    --wandb_project_name 'SSDLite320-MobileNetV3 Waste Classification' \
    --output_path 'validation_results'