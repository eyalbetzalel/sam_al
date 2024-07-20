import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

def polygon_to_mask(polygon, image_shape=(1024, 2048)):
    """
    Convert a polygon to a binary mask.

    Parameters:
    polygon (list): List of points defining the polygon.
    image_shape (tuple): Shape of the output mask.

    Returns:
    np.ndarray: Binary mask with the polygon filled in.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon)], 1)
    return mask

def calculate_rect_size(bbox):
    """
    Calculate the size of a rectangle.

    Parameters:
    bbox (torch.Tensor): Bounding box tensor.

    Returns:
    float: Size of the rectangle.
    """
    x1, y1, x2, y2 = bbox[0].cpu().numpy()
    width = x2 - x1
    height = y2 - y1
    return width * height

def get_values_from_data_iter(data, batch_size, predictor, input_image_size=(1024, 2048)):
    """
    Get values from data iterator.

    Parameters:
    data (list): List of data items.
    batch_size (int): Batch size.
    predictor (SamPredictor): SAM predictor.
    input_image_size (tuple): Input image size.

    Returns:
    tuple: Tuple containing ground truth masks, bounding boxes, and labels.
    """
    gt_mask = []
    for item in data:
        polygon = item['polygon']
        mask = polygon_to_mask(polygon)
        gt_mask.append(mask)
    gt_mask = torch.tensor(np.array(gt_mask)).to(predictor.device)

    bboxes = []
    for item in data:
        bbox = item['bbox']
        bboxes.append(bbox)
    bboxes = torch.tensor(np.array(bboxes), device=predictor.device)
    bboxes = predictor.transform.apply_boxes_torch(bboxes, input_image_size)

    labels = []
    for item in data:
        label = item['label']
        labels.append(label)
    
    gt_mask_split = torch.split(gt_mask, batch_size, dim=0)
    bboxes_split = torch.split(bboxes, batch_size, dim=0)
    labels_split = [labels[i:i+batch_size] for i in range(0, len(labels), batch_size)]
    
    return gt_mask_split, bboxes_split, labels_split

def visualize_and_save(image, gt_mask, sam_mask, curr_bbox, filename="test_sam.png"):
    """
    Visualize and save the image, ground truth mask, and SAM mask.

    Parameters:
    image (torch.Tensor): The original image.
    gt_mask (torch.Tensor): Ground truth mask.
    sam_mask (torch.Tensor): SAM mask.
    curr_bbox (torch.Tensor): Bounding box.
    filename (str): The filename to save the image to.

    Returns:
    None
    """
    gt_mask = gt_mask.detach().cpu().numpy()[0, :, :]
    sam_mask = sam_mask.detach().cpu().numpy()[0, :, :]
    curr_bbox = curr_bbox.detach().cpu().numpy()[0, :]
    image = image.detach().cpu().numpy().squeeze().transpose((1, 2, 0))

    gt_cmap = ListedColormap(['none', 'red'])
    sam_cmap = ListedColormap(['none', 'blue'])

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image)
    ax[0].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)
    ax[0].set_title('Image with GT Mask')
    min_x, min_y, max_x, max_y = curr_bbox * 2
    
    width = max_x - min_x
    height = max_y - min_y
    rect = patches.Rectangle((min_x, min_y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)

    ax[1].imshow(image)
    ax[1].imshow(sam_mask, cmap=sam_cmap, alpha=0.5)
    ax[1].set_title('Image with SAM Mask')

    ax[2].imshow(image)
    ax[2].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)
    ax[2].imshow(sam_mask, cmap=sam_cmap, alpha=0.5)
    ax[2].set_title('Image with GT and SAM Masks')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
