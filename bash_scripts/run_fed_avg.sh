#!/bin/bash

# Command-line arguments
n_clients=1
clients=("192.168.100.14:8080")
checkpoint="checkpoints/epoch_99.pth"
output_dir="output_dir"
port=8080
num_classes=5
learning_rate=0.001

# Call the Python script with arguments
python federated_training/federated_averaging.py  --n_clients $n_clients \
                        --clients "${clients[@]}" \
                        --checkpoint $checkpoint \
                        --output_dir $output_dir \
                        --port $port \
                        --num_classes $num_classes \
                        --learning_rate $learning_rate

