import argparse
import os
import pickle
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
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--reuse_images', action='store_true', default=False)
    return parser.parse_args()


def main(args):
    step = 0
    server_to_client_transfer_times = []
    client_to_server_transfer_times = []
    training_times = []
    inference_times = []

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
        transfer_time, transfer_time_2, inference_time, training_time = client_ft()
        server_to_client_transfer_times.append(transfer_time)
        client_to_server_transfer_times.append(transfer_time_2)
        training_times.append(training_time)
        inference_times.append(inference_time)
        print(f"Transfer time: {transfer_time:.2f} seconds, {transfer_time_2:.2f} seconds ")
        print(f"Training time: {training_time:.2f} seconds, Inference time: {inference_time:.2f} seconds ")
        step += 1
        if step > args.steps:
            break
        while os.listdir(args.input_img_directory) == image_set and not args.reuse_images:
            print("Waiting on new images...")
            time.sleep(10)
    with open('training_times.pkl', 'wb') as f:
        pickle.dump(training_times, f)
    with open('inference_times.pkl', 'wb') as f:
        pickle.dump(inference_times, f)
    with open('client_to_server_transfer_times.pkl', 'wb') as f:
        pickle.dump(client_to_server_transfer_times, f)
    with open('server_to_client_transfer_times.pkl', 'wb') as f:
        pickle.dump(server_to_client_transfer_times, f)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
