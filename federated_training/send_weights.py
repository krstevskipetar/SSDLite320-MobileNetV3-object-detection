import socket
import argparse
import os
import threading

from distributed_comms import send_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('remote_host', type=str)
    parser.add_argument('remote_port', type=int)
    parser.add_argument('file_path', type=str)

    return parser.parse_args()


def main(args):
    send_file(args.remote_host, args.remote_port, args.file_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
