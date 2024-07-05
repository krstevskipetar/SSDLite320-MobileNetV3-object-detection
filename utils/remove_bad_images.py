from torchvision.io import read_image
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='')
    return parser.parse_args()


def main(args):
    i = 0
    for filename in os.listdir(args.image_path):
        file_path = os.path.join(args.image_path, filename)
        try:
            image = read_image(file_path)
        except Exception as e:
            print(e)
            os.remove(file_path)
            i += 1
    print("Removed {} images".format(i))


if __name__ == '__main__':
    args = parse_args()
    main(args)
