import os
import shutil
import socket
import threading
import time

import torch

from model import get_model


class FedAvg:
    def __init__(self, n_clients: int = 0, clients: list[dict] = None, checkpoint: str = None,
                 output_dir: str = None, local_host: str = '127.0.0.1', port: int = 8080,
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
        self.global_model.load_state_dict(torch.load(self.checkpoint, map_location='cpu'))

        self.output_dir = output_dir
        self.client_index = 0

        self.local_host = local_host
        self.port = port

    def start_server(self, host, port, output_directory):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()

            print(f"Server listening on {host}:{port}")
            while True and self.client_index < self.n_clients:
                file_name = f'weights_{self.client_index}.pth'
                self.client_index += 1
                conn, addr = s.accept()
                print(f"Connected by {addr}")

                # Start a new thread to handle the connection
                client_thread = threading.Thread(target=self.handle_client, args=(conn, output_directory, file_name))
                client_thread.start()

    def handle_client(self, conn, output_directory, file_name):
        try:
            self.receive_file(conn, output_directory, file_name)
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
                self.send_file(client['address'], client['port'], self.checkpoint)

            self.start_server(self.local_host, self.port, self.output_dir)
            while self.client_index < self.n_clients:
                time.sleep(10)
            for checkpoint in os.listdir(self.output_dir):
                client_model = get_model(self.num_classes)
                client_model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
                self.server_update(client_model, self.learning_rate)

            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

            torch.save({
                'model_state_dict': self.global_model.state_dict(),
            }, self.checkpoint)
