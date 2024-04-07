import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from core.postprocessing import apply_nms


def plot_image(image_tensor, bounding_box, show_plot=True):
    image_np = image_tensor.cpu().permute(1, 2, 0).numpy()

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image_np)

    # Create a Rectangle patch
    x_min, y_min, x_max, y_max = bounding_box.cpu().int().tolist()
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    if show_plot:
        plt.show()

    return fig


def plot_image_multiple_boxes(image_tensor, bounding_boxes, labels, class_names, show_plot=True):
    image_np = image_tensor.cpu().permute(1, 2, 0).numpy()

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image_np)

    for bounding_box, label in zip(bounding_boxes, labels):
        # Create a Rectangle patch
        x_min, y_min, x_max, y_max = bounding_box.cpu().int().tolist()
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='g', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(x_min, y_min, f'{class_names[label]}', bbox=dict(facecolor='white', alpha=0.5))

    if show_plot:
        plt.show()

    return fig


def plot_predictions(image, prediction, gt=None, class_names=None, show_plot=True, nms_threshold: float = 0.1,
                     score_threshold: float = 0.2):
    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']

    keep = apply_nms(boxes, scores, nms_threshold, score_threshold)
    boxes = boxes.tolist()
    scores = scores.tolist()
    labels = labels.tolist()

    # Plot the image
    fig, ax = plt.subplots(1)
    image = np.array(image)
    image = image.transpose(1, 2, 0)
    ax.imshow(image)

    # Add bounding boxes to the image
    for i in keep:
        box = boxes[i]
        score = scores[i]
        label = class_names[labels[i]]

        # Create a Rectangle patch
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add label and score to the bounding box
        ax.text(box[0], box[1], f'{label}: {score:.2f}', bbox=dict(facecolor='white', alpha=0.5))
    if gt:
        boxes_gt = gt['boxes'].cpu()
        labels_gt = gt['labels'].cpu()
        for box, label in zip(boxes_gt, labels_gt):
            label = class_names[label]
            # Create a Rectangle patch
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='g',
                                     facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add label and score to the bounding box
            ax.text(box[0], box[1], f'{label}: {1:.2f}', bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    if show_plot:
        plt.show()
    return fig


def plot_predictions_in_grid(images, predictions, ground_truths, class_names, n_cols=3, show_plot=True):
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    for i in range(n_rows):
        for j in range(n_cols):
            index = i * n_cols + j
            if index < n_images:
                image = images[index]
                prediction = predictions[index]
                gt = ground_truths[index]

                boxes = prediction['boxes']
                scores = prediction['scores']
                labels = prediction['labels']

                boxes_gt = gt['boxes'].cpu()
                labels_gt = gt['labels'].cpu()

                keep = apply_nms(boxes, scores, 0.1)
                boxes = boxes.tolist()
                scores = scores.tolist()
                labels = labels.tolist()

                # Plot the image
                ax = axs[i, j] if n_rows > 1 else axs[j]
                image = np.array(image) * 255
                image = image.astype(np.uint8)
                image = image.transpose(1, 2, 0)
                ax.imshow(image)

                # Add bounding boxes to the image
                for k in keep:
                    box = boxes[k]
                    score = scores[k]
                    label = class_names[labels[k]]

                    # Create a Rectangle patch
                    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                             linewidth=1, edgecolor='r', facecolor='none')

                    # Add the patch to the Axes
                    ax.add_patch(rect)

                    # Add label and score to the bounding box
                    ax.text(box[0], box[1], f'{label}: {score:.2f}', bbox=dict(facecolor='white', alpha=0.5))

                labels_gt = labels_gt.tolist()
                for box, label in zip(boxes_gt, labels_gt):
                    label = class_names[label]
                    # Create a Rectangle patch
                    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                             linewidth=1, edgecolor='g', facecolor='none')

                    # Add the patch to the Axes
                    ax.add_patch(rect)

                    # Add label and score to the bounding box
                    ax.text(box[0], box[1], f'{label}: {1:.2f}', bbox=dict(facecolor='white', alpha=0.5))

                ax.axis('off')

    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig


def calculate_dimensions(num_plots):
    if num_plots <= 1:
        n_rows = 1
        n_columns = num_plots
    else:
        n_rows = 2
        n_columns = (num_plots + 1) // n_rows

    return n_rows, n_columns


def plot_metrics(metrics: dict):
    n_rows, n_columns = calculate_dimensions(len(metrics.keys()))
    fig, axs = plt.subplots(n_rows, n_columns, layout="constrained")

    for key, ax in zip(metrics.keys(), axs):
        metric = metrics[key]
        for k in metric.keys():
            ax.scatter(k, metric[k])
        values = [metric[k] for k in metric.keys()]
        max_val = np.max(values), np.argmax(values) * 5

        ax.text(max_val[1], max_val[0] - 0.06, f'{key}: {max_val[0]:.4f}\n Epoch: {max_val[1]}')
        values = np.interp([i for i in range(np.max([int(k) for k in metric.keys()]))],
                           [int(k) for k in metric.keys()], values)
        ax.plot(values)
        ax.set_title(key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)

    plt.show()
    return fig
