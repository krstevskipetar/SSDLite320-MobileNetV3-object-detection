from yolo_dataset import YOLODataset
import torch
import argparse
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='G:\\waste_dataset_v2\\train\\images')
    parser.add_argument('--annotation_path', default='G:\\waste_dataset_v2\\train\\labels')
    parser.add_argument('--label_file', default='G:\\waste_dataset_v2\\label_map.txt')
    parser.add_argument('--shuffle', action='store_true', default=False)
    argz = parser.parse_args()

    return argz


args = parse_args()

dataset = YOLODataset(image_path=args.image_path,
                      annotation_path=args.annotation_path,
                      label_file=args.label_file,
                      shuffle=args.shuffle)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=torch.utils.data.collate_fn
)

model = get_model()

images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)  # Returns losses and detections
print(output)


def main():
    pass


if __name__ == "__main__":
    main()
