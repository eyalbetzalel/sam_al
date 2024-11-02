import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from torch.optim.lr_scheduler import _LRScheduler

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
    x1, y1, x2, y2 = bbox
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
        mask = item['mask']
        gt_mask.append(mask)

    if len(gt_mask) == 0:
        return None, None, None
    
    gt_mask = torch.stack(gt_mask, dim=0).to(predictor.device)

    keep_indices = []
    bboxes = []
    for i, item in enumerate(data):
        bbox = item['bbox']
        bboxes.append(bbox)
        if calculate_rect_size(bbox) > 10000 and calculate_rect_size(bbox) < 1048576:
            keep_indices.append(i)
    bboxes = torch.stack(bboxes, dim=0).to(predictor.device)
    bboxes = predictor.transform.apply_boxes_torch(bboxes, input_image_size)

    labels = []
    for item in data:
        label = item['category']
        labels.append(label)
    
    bboxes = bboxes[keep_indices]
    gt_mask = gt_mask[keep_indices]
    labels = [labels[i] for i in keep_indices]
    
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
    # ax[0].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)
    ax[0].set_title('Image')


    ax[1].imshow(image)
    ax[1].imshow(sam_mask, cmap=sam_cmap, alpha=0.5)
    ax[1].set_title('Image with SAM Mask')

    ax[2].imshow(image)
    ax[2].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)
    ax[2].imshow(sam_mask, cmap=sam_cmap, alpha=0.5)
    ax[2].set_title('Image with GT and SAM Masks')
    min_x, min_y, max_x, max_y = curr_bbox * 2
    
    width = max_x - min_x
    height = max_y - min_y
    rect = patches.Rectangle((min_x, min_y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax[2].add_patch(rect)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, initial_lr, final_lr, last_epoch=-1):
        """
        Custom scheduler to warm up the learning rate.

        Parameters:
        optimizer (Optimizer): Optimizer to apply the warmup to.
        warmup_steps (int): Number of steps for the warmup phase.
        initial_lr (float): Starting learning rate for the warmup.
        final_lr (float): Final learning rate after warmup.
        last_epoch (int): The index of the last epoch.
        """
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_step = (final_lr - initial_lr) / warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linearly increase the learning rate
            return [self.initial_lr + self.lr_step * self.last_epoch for _ in self.base_lrs]
        else:
            # Return the base learning rate (final_lr) after warmup
            return [self.final_lr for _ in self.base_lrs]