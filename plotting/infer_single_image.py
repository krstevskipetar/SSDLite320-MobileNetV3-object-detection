import argparse

import torch
import torchvision.io

import data.load_data
from core.model import get_model
from core.plotting import plot_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img')
    parser.add_argument('checkpoint')
    parser.add_argument('label_file')
    parser.add_argument('--nms_iou_threshold', type=float, default=0.5)
    parser.add_argument('--score_threshold', type=float, default=0.2)

    return parser.parse_args()


def main(args):
    resize = torchvision.transforms.Resize(size=(320, 320), antialias=True)
    model = get_model(num_classes=5)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    model.eval()

    img = torchvision.io.read_image(args.img)
    img = resize(img)
    predictions = model([img.float() / 255])
    class_names = data.load_data.load_class_names(args.label_file)
    class_names = {i: c for (i, c) in zip(range(1, len(class_names)), class_names)}
    plot_predictions(image=img, prediction=predictions[0], class_names=class_names,
                     nms_threshold=args.nms_iou_threshold,
                     score_threshold=args.score_threshold)


if __name__ == "__main__":
    args = parse_args()
    main(args)
