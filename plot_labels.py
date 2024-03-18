import argparse
import os
from os.path import join

import matplotlib.pyplot as plt
import torch
import torchvision.io
from tqdm import tqdm
from core.plotting import plot_image_multiple_boxes

from data.load_data import load_dataset, create_dataloader, load_class_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='/home/petar/waste_dataset_v2/inference/images')
    parser.add_argument('--annotation_path', default='/home/petar/waste_dataset_v2/inference/labels')
    parser.add_argument('--label_file', default='/home/petar/waste_dataset_v2/label_map.txt')
    parser.add_argument('--write_to', default='/home/petar/waste_dataset_v2/inference/images_inferred')
    parser.add_argument('--plot', action='store_true', default=False)
    return parser.parse_args()


def plot_labels(image_path, annotation_path, label_file, write_to, plot):
    if write_to:
        os.makedirs(write_to, exist_ok=True)
    dataset = load_dataset(image_path, annotation_path, label_file=label_file, shuffle=False)
    dataloader = create_dataloader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    class_names = load_class_names(args.label_file)
    print(class_names)
    class_names = {i: c for (i, c) in zip(range(1, len(class_names) + 1), class_names)}
    print(class_names)
    for idx, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True):
        image = images[0]
        bboxes = targets[0]['boxes']
        labels = targets[0]['labels'].tolist()
        fig = plot_image_multiple_boxes(image, bboxes, labels, class_names, show_plot=plot)
        if write_to:
            fig.savefig('{}/{}.png'.format(write_to, idx))
        plt.close(fig)


if __name__ == '__main__':
    args = parse_args()
    plot_labels(args.image_path, args.annotation_path, args.label_file, args.write_to, args.plot)
