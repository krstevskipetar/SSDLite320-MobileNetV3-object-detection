import os
import socket
import threading
import time

import torch

from engine import train_epoch
from model import get_model


class ClientFineTune:
    def __init__(self, checkpoint=None, num_classes=5, optimizer=None,
                 lr_scheduler=None, data_loader=None,
                 device='cpu', server_address=None, server_port=None):
        self.model = get_model(num_classes)
        self.checkpoint = checkpoint
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=device)['model_state_dict'])
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader
        self.device = device

        self.server_address = server_address
        self.server_port = server_port

    def send_file(self, host, port, file_path):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            self.send_file_data(s, file_path)
            s.close()

    def send_file_data(self, conn, file_path):
        file_size = os.path.getsize(file_path)
        conn.sendall(str(file_size).encode('utf-8'))
        time.sleep(5)
        print(f"Sending file of size {file_size} bytes")

        with open(file_path, 'rb') as file:
            while True:
                data = file.read(1024)
                if not data:
                    break
                conn.sendall(data)

        print("File sent successfully")

    def receive_file(self, conn, output_directory, file_name):
        file_name = os.path.join(output_directory, file_name)
        data = conn.recv(1024)
        file_size = int(data.decode('utf-8'))

        print(f"Receiving file of size {file_size} bytes")

        with open(file_name, 'wb') as file:
            while file_size > 0:
                file.flush()
                data = conn.recv(1024)
                file.write(data)
                file.flush()
                file_size -= len(data)
                print(f"Received {len(data)} bytes, remaining {file_size} bytes")

        print("File received successfully")

    def receive_from_server(self, host, port, output_directory):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            print(f"Listening on {host}:{port}")
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            self.receive_file(conn, output_directory, self.checkpoint)

    def handle_client(self, conn, output_directory, file_name):
        try:
            self.receive_file(conn, output_directory, file_name)
        finally:
            conn.close()

    def __call__(self, *args, **kwargs):
        while True:
            print(f"Waiting for checkpoint from server at {self.server_address}:{self.server_port}...")

            self.receive_from_server(self.server_address, self.server_port, self.checkpoint)
            self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device)['model_state_dict'])

            print("Checkpoint loaded, training one epoch...")
            train_epoch(self.model, self.optimizer, self.lr_scheduler,
                        self.data_loader, self.device)
            torch.save({
                'model_state_dict': self.model.state_dict(),
            }, self.checkpoint)

            print(f"Training complete, sending checkpoint to server at {self.server_address}:{self.server_port}")
            self.send_file(self.server_address, self.server_port, self.checkpoint)
