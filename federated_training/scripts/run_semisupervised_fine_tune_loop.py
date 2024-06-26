import argparse
import os
import time

from federated_training.client_finetuning import ClientFineTune


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_directory', required=True)
    parser.add_argument('--input_img_directory', required=True)
    parser.add_argument('--output_annotation_directory', required=True)
    parser.add_argument('--label_file', required=True)
    parser.add_argument('--server_address', required=True)
    parser.add_argument('--server_port', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--score_threshold', type=float, default=0.2)
    return parser.parse_args()


def main(args):
    while True:
        print(os.listdir(args.input_img_directory))
        while len(os.listdir(args.input_img_directory)) == 0:
            print("No images yet, waiting.")
            time.sleep(10)
        image_set = os.listdir(args.input_img_directory)

        client_ft = ClientFineTune(image_path=args.input_img_directory,
                                   annotation_path=args.output_annotation_directory,
                                   device=args.device, num_classes=args.num_classes,
                                   checkpoint_directory=args.checkpoint_directory,
                                   label_file=args.label_file,
                                   server_address=args.server_address,
                                   server_port=args.server_port)
        client_ft()

        while os.listdir(args.input_img_directory) == image_set:
            print("Waiting on new images...")
            time.sleep(10)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
