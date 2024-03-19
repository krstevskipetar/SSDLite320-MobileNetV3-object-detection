import torch

from vision.references.detection.utils import collate_fn
from data.yolo_dataset import YOLODataset


def load_train_and_val_datasets(image_path_train, annotation_path_train, image_path_val, annotation_path_val,
                                label_file=None,
                                shuffle_train=False, shuffle_val=False):
    dataset_train = load_dataset(image_path=image_path_train,
                                 annotation_path=annotation_path_train,
                                 label_file=label_file,
                                 shuffle=shuffle_train)

    dataset_val = load_dataset(image_path=image_path_val,
                               annotation_path=annotation_path_val,
                               label_file=label_file,
                               shuffle=shuffle_val)
    return dataset_train, dataset_val


def load_dataset(image_path, annotation_path, label_file, shuffle):
    dataset = YOLODataset(image_path=image_path,
                          annotation_path=annotation_path,
                          label_file=label_file,
                          shuffle=shuffle)
    return dataset


def create_train_and_val_dataloaders(dataset_train, dataset_val, batch_size_train, batch_size_val, shuffle_train,
                                     shuffle_val, num_workers):
    data_loader_train = create_dataloader(dataset_train, batch_size_train, shuffle_train, num_workers)
    data_loader_val = create_dataloader(dataset_val, batch_size_val, shuffle_val, num_workers)
    return data_loader_train, data_loader_val


def create_dataloader(dataset, batch_size, shuffle, num_workers):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return data_loader


def load_class_names(label_file):
    with open(label_file, 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    return classes
