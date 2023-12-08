import socket
import argparse
import os
import threading

from distributed_comms import send_file, start_server


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('local_host', type=str)
    parser.add_argument('local_port', type=int)
    parser.add_argument('output_directory', type=str)

    return parser.parse_args()


def main(args):
    server_thread = threading.Thread(target=start_server, args=(args.local_host, args.local_port,
                                                                args.output_directory))
    server_thread.start()


if __name__ == "__main__":
    args = parse_args()
    main(args)
