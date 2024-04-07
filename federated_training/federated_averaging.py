import argparse
import os
import shutil
import socket
import sys
import time
from os.path import join

import torch

from core.model import get_model
from federated_training.distributed_comms import send_file, receive_file


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
    def __init__(self, n_clients: int = 0,
                 clients: list[dict] = None,
                 device: str = 'cpu',
                 checkpoint: str = None,
                 output_dir: str = None,
                 port: int = 8080,
                 num_classes: int = 5,
                 learning_rate: float = 0.001):
        """

        :param n_clients: Number of clients
        :param clients: Information for clients, should contain 'address' and 'port' keys
        """

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.n_clients = n_clients
        self.clients = clients
        self.device = device

        self.checkpoint = checkpoint
        self.global_model = get_model(num_classes=num_classes)
        self.global_model.load_state_dict(torch.load(self.checkpoint, map_location='cpu')['model_state_dict'])

        self.output_dir = output_dir
        self.client_index = 0

        self.port = port
        self.receiving_socket = None

    def receive_client_weights(self):
        if self.receiving_socket is None:
            try:
                self.receiving_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.receiving_socket.bind(('0.0.0.0', self.port))
                self.receiving_socket.listen()
            except socket.error as e:
                print('Socket creation failed: {}'.format(e))
                sys.exit(socket.SO_ERROR)

        print(f"Server listening on 0.0.0.0:{self.port}")
        while True and self.client_index < self.n_clients:
            file_name = f'weights_{self.client_index}.pth'
            conn, addr = self.receiving_socket.accept()
            print(f"Connected by {addr}")
            self.handle_client(conn, self.output_dir, file_name)

    def handle_client(self, conn, output_directory, file_name):
        try:
            receive_file(conn, output_directory, file_name)
            self.client_index += 1
        finally:
            conn.close()

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
                        send_file(client['address'], client['port'], self.checkpoint)
                        connected = True
                    except ConnectionRefusedError as e:
                        print(e)
                        print("Client not available, waiting...")
                        time.sleep(15)
            self.receive_client_weights()

            for checkpoint in os.listdir(self.output_dir):
                client_model = get_model(self.num_classes)
                checkpoint = torch.load(join(self.output_dir, checkpoint), map_location=self.device)
                client_model.load_state_dict(checkpoint['model_state_dict'])
                self.server_update(client_model, self.learning_rate)

            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

            torch.save({
                'model_state_dict': self.global_model.state_dict(),
            }, join('local_data', 'global_model.pth'))
            self.client_index = 0


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
