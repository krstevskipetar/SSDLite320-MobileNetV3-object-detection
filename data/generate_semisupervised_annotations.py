import argparse
import os
import shutil
import time
from os.path import join
import random

import torch
import torchvision.io
from tqdm import tqdm

from core.model import get_model
from core.postprocessing import apply_nms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('input_directory')
    parser.add_argument('output_directory')
    parser.add_argument('--infinite', action='store_true')
    parser.add_argument('--sample_directory', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--max_files', type=int, default=1000)
    parser.add_argument('--pause_time', type=int, default=60,
                        help='Time to pause between sampling/inference steps, only valid when infinite=True')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--score_threshold', type=float, default=0.8)
    return parser.parse_args()


def infer_annotations(checkpoint, input_directory, output_directory, device='cpu', num_classes=5, iou_threshold=0.5,
                      score_threshold=0.2):
    def bbox2yolobbox(bbox):
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = (x2 - x1)
        h = (y2 - y1)
        return x / 320, y / 320, w / 320, h / 320

    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state_dict'])
    model.eval()
    image_loader = ((img_name, torchvision.io.read_image(join(input_directory, img_name))) for img_name in
                    os.listdir(input_directory))
    resize = torchvision.transforms.Resize(size=(320, 320), antialias=True)
    for img_name, img in tqdm(image_loader):
        img = resize(img)
        torchvision.io.write_png(img, join(input_directory, img_name))  # write resized image

        predictions = model([img.float() / 255])[0]

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
                box = bbox2yolobbox(box)
                f.write(f'{label} {box[0]} {box[1]} {box[2]} {box[3]}\n')


def main(args):
    if args.sample_directory is not None and len(os.listdir(args.input_directory)) < args.max_files:
        sampled_files = random.sample(os.listdir(args.sample_directory), args.n_samples)
        for file in sampled_files:
            shutil.copy(join(args.sample_directory, file), join(args.input_directory, file))
    infer_annotations(args.checkpoint, args.input_directory, args.output_directory, args.device, args.num_classes,
                      args.iou_threshold, args.score_threshold)


if __name__ == "__main__":
    args = parse_args()
    if args.infinite:
        while True:
            main(args)
            time.sleep(args.pause_time)
    else:
        main(args)
