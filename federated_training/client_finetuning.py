import os
import socket
import time

import torch

from core.engine import train_epoch
from core.model import get_model
from data.generate_semisupervised_annotations import infer_annotations
from data.load_data import create_dataloader
from data.yolo_dataset import YOLODataset
from federated_training.distributed_comms import send_file, receive_file
from time import perf_counter


class ClientFineTune:
    def __init__(self, image_path: str,
                 annotation_path: str,
                 label_file: str,
                 checkpoint_directory=None,
                 num_classes=5,
                 device='cpu',
                 batch_size=2,
                 learning_rate=0.0001,
                 server_address=None,
                 server_port=None,
                 local_address='0.0.0.0',
                 local_port=8080):
        self.model = get_model(num_classes)
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint = 'checkpoint.pth'
        self.local_address = local_address
        self.local_port = local_port

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )

        self.device = device
        self.batch_size = batch_size

        self.image_directory = image_path
        self.annotation_directory = annotation_path
        self.num_classes = num_classes
        self.label_file = label_file

        self.server_address = server_address
        self.server_port = server_port

    def receive_from_server(self, host, port, output_directory):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            print(f"Listening on {host}:{port}")
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            receive_file(conn, output_directory, self.checkpoint)

    def handle_client(self, conn, output_directory, file_name):
        try:
            receive_file(conn, output_directory, file_name)
        finally:
            conn.close()

    def __call__(self, *args, **kwargs):
        print(os.path.abspath(self.image_directory))
        while len(os.listdir(self.image_directory)) == 0:
            print("No images yet!")
            time.sleep(10)
            continue

        start_transfer = perf_counter()
        print(f"Waiting for checkpoint from server at {self.server_address}:{self.server_port}...")
        self.receive_from_server(self.local_address, self.local_port, self.checkpoint_directory)
        end_transfer = perf_counter()
        transfer_time = end_transfer - start_transfer

        start_inference = perf_counter()
        # infer annotations for input images
        infer_annotations(checkpoint=os.path.join(self.checkpoint_directory, self.checkpoint),
                          input_directory=self.image_directory,
                          output_directory=self.annotation_directory,
                          device=self.device,
                          num_classes=self.num_classes,
                          iou_threshold=0.1,
                          score_threshold=0.7)
        end_inference = perf_counter()
        inference_time = end_inference - start_inference
        start_training = perf_counter()
        dataset = YOLODataset(image_path=self.image_directory,
                              annotation_path=self.annotation_directory,
                              label_file=self.label_file)
        dataset._validate_input()
        self.data_loader = create_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1
        )

        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_directory, self.checkpoint),
                                              map_location=self.device)['model_state_dict'])

        print("Checkpoint loaded, training one epoch...")
        all_losses, mean_loss = train_epoch(model=self.model,
                                            optimizer=self.optimizer,
                                            data_loader=self.data_loader,
                                            device=self.device)

        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, self.checkpoint)
        end_training = perf_counter()
        training_time = end_training - start_training
        print(f"Training complete, sending checkpoint to server at {self.server_address}:{self.server_port}")
        start_transfer_2 = perf_counter()
        send_file(self.server_address, self.server_port, self.checkpoint)
        end_transfer_2 = perf_counter()
        transfer_time_2 = end_transfer_2 - start_transfer_2
        return transfer_time, transfer_time_2, inference_time, training_time, all_losses, mean_loss
