#!/bin/bash

CHECKPOINT_PATH="local_data/checkpoints/checkpoint.pth"
INPUT_DIRECTORY="local_data/images"
OUTPUT_DIRECTORY="local_data/annotations"
SAMPLE_DIRECTORY="/home/petar/waste_dataset_v2/test/images"
N_SAMPLES=100
MAX_FILES=1000
PAUSE_TIME=1200
DEVICE="cpu"
NUM_CLASSES=5
IOU_THRESHOLD=0.5
SCORE_THRESHOLD=0.8
INFINITE=false
IGNORE_ANNOTATED=true
# Check if the infinite flag is provided
if [ "$1" == "--infinite" ]; then
    INFINITE=true
fi

CMD="python data/generate_semisupervised_annotations.py $CHECKPOINT_PATH $INPUT_DIRECTORY $OUTPUT_DIRECTORY \
    --sample_directory $SAMPLE_DIRECTORY --n_samples $N_SAMPLES --max_files $MAX_FILES \
    --pause_time $PAUSE_TIME --device $DEVICE --num_classes $NUM_CLASSES \
    --iou_threshold $IOU_THRESHOLD --score_threshold $SCORE_THRESHOLD"

# Append the --infinite flag if necessary
if [ "$INFINITE" = true ]; then
    CMD="$CMD --infinite"
fi
if [ "$IGNORE_ANNOTATED" = true ]; then
    CMD="$CMD --ignore_annotated"
fi
# Run the command
eval $CMD
