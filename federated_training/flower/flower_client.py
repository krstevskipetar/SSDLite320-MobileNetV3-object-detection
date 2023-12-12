import argparse
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import flwr as fl

from core.model import get_model
from vision.references.detection.engine import train_epoch, validate
from vision.references.detection.utils import collate_fn
from data.yolo_dataset import YOLODataset


def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, data_loader, data_loader_val, device='cpu'):
        self.model = model
        self.data_loader = data_loader
        self.data_loader_val = data_loader_val
        self.device = torch.device(device)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params,
            lr=0.001,
            momentum=0.9,
            weight_decay=0.0005
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                       T_max=10,
                                                                       eta_min=0)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        all_losses, epoch_loss = train_epoch(model=self.model,
                                             optimizer=self.optimizer,
                                             lr_scheduler=self.lr_scheduler,
                                             data_loader=self.data_loader,
                                             print_freq=1e12,
                                             epoch_number=0,
                                             all_losses={})
        return get_parameters(self.model), len(self.data_loader)

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = validate(self.model, self.data_loader_val, self.device)
        return float(loss), len(self.data_loader_val), {"accuracy": float(accuracy)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='/home/petar/waste_dataset_v2/train/images')
    parser.add_argument('--annotation_path', default='/home/petar/waste_dataset_v2/train/labels')
    parser.add_argument('--image_path_val', default='/home/petar/waste_dataset_v2/val/images')
    parser.add_argument('--annotation_path_val', default='/home/petar/waste_dataset_v2/val/labels')
    parser.add_argument('--label_file', default='/home/petar/waste_dataset_v2/label_map.txt')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default=None)

    argz = parser.parse_args()

    return argz


if __name__ == "__main__":
    args = parse_args()
    dataset = YOLODataset(image_path=args.image_path,
                          annotation_path=args.annotation_path,
                          label_file=args.label_file,
                          shuffle=args.shuffle)

    dataset_val = YOLODataset(image_path=args.image_path_val,
                              annotation_path=args.annotation_path_val,
                              label_file=args.label_file,
                              shuffle=args.shuffle)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        # Subset(dataset, random.sample([i for i in range(len(dataset))], 100)),
        batch_size=2,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
        drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        # Subset(dataset, random.sample([i for i in range(len(dataset_val))], 100)),
        batch_size=2,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        drop_last=True
    )

    model = get_model(num_classes=4, trainable_backbone_layers=6, weights=None)
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=torch.device(args.device))['model_state_dict'])

    fl.client.start_numpy_client(server_address="[::]:8080", client=FlowerClient(model=model,
                                                                                 data_loader=data_loader,
                                                                                 data_loader_val=data_loader_val,
                                                                                 device=args.device))
