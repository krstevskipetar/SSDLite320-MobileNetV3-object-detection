import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision

from postprocessing import apply_nms


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


def plot_predictions(image, prediction, gt, class_names=None, show_plot=True):
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
