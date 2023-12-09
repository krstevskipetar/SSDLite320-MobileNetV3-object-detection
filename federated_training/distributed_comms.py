import os
import threading
import socket


def start_server(host, port, output_directory):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        client_index = 0
        s.bind((host, port))
        s.listen()

        print(f"Server listening on {host}:{port}")
        while True:
            file_name = f'weights_{client_index}.pth'
            client_index += 1
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            # Start a new thread to handle the connection
            client_thread = threading.Thread(target=handle_client, args=(conn, output_directory, file_name))
            client_thread.start()


def handle_client(conn, output_directory, file_name):
    try:
        receive_file(conn, output_directory, file_name)
    finally:
        conn.close()


def send_file(host, port, file_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        send_file_data(s, file_path)


def receive_file(conn, output_directory, file_name):
    file_name = os.path.join(output_directory, file_name)
    file_size = int(conn.recv(1024).decode('utf-16'))
    print(f"Receiving file of size {file_size} bytes")

    with open(file_name, 'wb') as file:
        while file_size > 0:
            data = conn.recv(1024)
            file.write(data)
            file_size -= len(data)
            print(f"Received {len(data)} bytes, remaining {file_size} bytes")

    print("File received successfully")


def send_file_data(conn, file_path):
    file_size = os.path.getsize(file_path)
    conn.sendall(str(file_size).encode('utf-16'))
    print(f"Sending file of size {file_size} bytes")

    with open(file_path, 'rb') as file:
        while True:
            data = file.read(1024)
            if not data:
                break
            conn.sendall(data)

    print("File sent successfully")
