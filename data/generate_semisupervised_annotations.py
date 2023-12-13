import argparse

import torch
import torchvision.io

from core.model import get_model
import os
from os.path import join

from core.postprocessing import apply_nms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('input_directory')
    parser.add_argument('output_directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--iou_threshold', type=float, default=0.1)
    parser.add_argument('--score_threshold', type=float, default=0.2)


def infer_annotations(checkpoint, input_directory, output_directory, device='cpu', num_classes=5, iou_threshold=0.5,
                      score_threshold=0.2):
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    image_loader = ((img_name, torchvision.io.read_image(join(input_directory, img_name))) for img_name in
                    os.listdir(input_directory))
    resize = torchvision.transforms.Resize(size=(320, 320), antialias=True)
    for img_name, img in image_loader:
        img = resize(img)
        torchvision.io.write_png(img, join(input_directory, img_name))  # write resized image

        predictions = model([img.float() / 255])

        keep = apply_nms(predictions['boxes'], predictions['scores'], threshold=iou_threshold,
                         score_threshold=score_threshold)
        kept_predictions = {'boxes': predictions['boxes'][keep],
                            'labels': predictions['labels'][keep]}

        if device != 'cpu':
            kept_predictions['boxes'] = [box.cpu() for box in kept_predictions['boxes']]
            kept_predictions['labels'] = [label.cpu() for label in kept_predictions['labels']]
        file_name = join(output_directory, img_name.split('.')[0] + '.txt')
        with open(file_name, 'w') as f:
            for pred_box, pred_label in zip(kept_predictions['boxes'], kept_predictions['labels']):
                label = int(pred_label)
                box = pred_box.tolist()
                f.write(f'{label} {box[0]} {box[1]} {box[2]} {box[3]}\n')


def main(args):
    infer_annotations(args.checkpoint, args.input_directory, args.output_directory, args.device, args.num_classes,
                      args.iou_threshold, args.score_threshold)
