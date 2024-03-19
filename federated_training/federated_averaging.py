import argparse
import os
import shutil
import socket
import threading
import time
from os.path import join

import torch

from core.model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for FedAvg class")
    parser.add_argument("--n_clients", type=int, default=0, help="Number of clients")
    parser.add_argument("--clients", nargs='+', type=str, default=None, help="List of clients")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    return parser.parse_args()


class FedAvg:
    def __init__(self, n_clients: int = 0, clients: list[dict] = None, checkpoint: str = None,
                 output_dir: str = None, port: int = 8080,
                 num_classes: int = 5, learning_rate: float = 0.001):
        """

        :param n_clients: Number of clients
        :param clients: Information for clients, should contain 'address' and 'port' keys
        """

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.n_clients = n_clients
        self.clients = clients

        self.checkpoint = checkpoint
        self.global_model = get_model(num_classes=num_classes)
        self.global_model.load_state_dict(torch.load(self.checkpoint, map_location='cpu')['model_state_dict'])

        self.output_dir = output_dir
        self.client_index = 0

        self.port = port

    def start_server(self, host, port, output_directory):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()

            print(f"Server listening on {host}:{port}")
            while True and self.client_index < self.n_clients:
                file_name = f'weights_{self.client_index}.pth'
                conn, addr = s.accept()
                print(f"Connected by {addr}")
                self.handle_client(conn, output_directory, file_name)

    def handle_client(self, conn, output_directory, file_name):
        try:
            self.receive_file(conn, output_directory, file_name)
            self.client_index += 1
        finally:
            conn.close()

    def send_file(self, host, port, file_path):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            self.send_file_data(s, file_path)
            s.close()

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

    def server_update(self, model, learning_rate):
        with torch.no_grad():
            for param, global_param in zip(model.parameters(), self.global_model.parameters()):
                update = learning_rate * (param.data - global_param.data)
                global_param.data.add_(update)

    def __call__(self, *args, **kwargs):
        while True:
            for client_idx, client in enumerate(self.clients):
                print(f"Sending global model to {client['address']}:{client['port']}")
                connected = False
                while not connected:
                    try:
                        self.send_file(client['address'], client['port'], self.checkpoint)
                        connected = True
                    except ConnectionRefusedError as e:
                        print(e)
                        print("Client not available, waiting...")
                        time.sleep(15)

            self.start_server('0.0.0.0', self.port, self.output_dir)
            while self.client_index < self.n_clients:
                time.sleep(10)
            for checkpoint in os.listdir(self.output_dir):
                client_model = get_model(self.num_classes)
                client_model.load_state_dict(torch.load(join(self.output_dir, checkpoint), map_location='cpu'))
                self.server_update(client_model, self.learning_rate)

            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

            torch.save({
                'model_state_dict': self.global_model.state_dict(),
            }, join('local_data', 'global_model.pth'))


if __name__ == "__main__":
    args = parse_args()
    clients = []
    for client in args.clients:
        address, port = client.split(':')
        clients.append({'address': address, 'port': int(port)})
    print(clients)
    fed_avg = FedAvg(n_clients=args.n_clients, clients=clients, checkpoint=args.checkpoint,
                     output_dir=args.output_dir, port=args.port,
                     num_classes=args.num_classes, learning_rate=args.learning_rate)
    fed_avg()
