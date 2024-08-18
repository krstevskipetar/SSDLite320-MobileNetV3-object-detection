#!/bin/bash

n_clients=1
clients=("192.168.100.67:8080", "192.168.100.68:8080")
checkpoint="local_data/checkpoints/checkpoint.pth"
output_dir="output_dir"
port=8080
num_classes=5
learning_rate=0.001
image_path_val='/home/wfpetarkrestevski/Desktop/waste_dataset_v2/val/images'
annotation_path_val='/home/wfpetarkrestevski/Desktop/waste_dataset_v2/val/labels'
label_file='/home/wfpetarkrestevski/Desktop/waste_dataset_v2/label_map.txt'
steps=10
python federated_training/federated_averaging.py  --n_clients $n_clients \
                        --clients "${clients[@]}" \
                        --checkpoint $checkpoint \
                        --output_dir $output_dir \
                        --port $port \
                        --num_classes $num_classes \
                        --learning_rate $learning_rate \
                        --image_path_val $image_path_val \
                        --annotation_path_val $annotation_path_val \
                        --label_file $label_file \
                        --steps $steps \
                        --validate

