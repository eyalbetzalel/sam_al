

# Fine tune the model with the newly annotated instances : 

# https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb
# https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/


import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 
from importCityScapesToDataloader import train_dataset, val_dataset, test_dataset
from segment_anything import sam_model_registry, SamPredictor
from matplotlib.colors import ListedColormap
from samplingUtils import ActiveLearningDataset
import torch 
import matplotlib.patches as patches
import torch.nn as nn

def visualize_and_save(image, gt_mask, sam_mask, curr_bbox, filename="test_sam.png"):
    """
    Visualize and save the image, ground truth mask, and SAM mask.

    Parameters:
    image (numpy.ndarray): The original image.
    gt_mask (numpy.ndarray): Ground truth mask.
    sam_mask (numpy.ndarray): SAM mask.
    filename (str): The filename to save the image to.

    Returns:
    None
    """

    # Preprocess (deteach, to cpu, to numpy): 
    gt_mask = gt_mask.detach().cpu().numpy()[0,:,:]
    sam_mask = sam_mask.detach().cpu().numpy()[0,:,:]
    curr_bbox = curr_bbox.detach().cpu().numpy()[0,:]
    image = image.detach().cpu().numpy().squeeze().transpose((1, 2, 0))

    # Create custom colormaps
    gt_cmap = ListedColormap(['none', 'red'])
    sam_cmap = ListedColormap(['none', 'blue'])

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image)
    ax[0].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)  # Overlay GT mask with opacity
    ax[0].set_title('Image with GT Mask')
    min_x, min_y, max_x, max_y = curr_bbox * 2
    
    width = max_x - min_x
    height = max_y - min_y
    rect = patches.Rectangle((min_x, min_y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)

    ax[1].imshow(image)
    ax[1].imshow(sam_mask, cmap=sam_cmap, alpha=0.5)  # Overlay SAM mask with opacity
    ax[1].set_title('Image with SAM Mask')

    ax[2].imshow(image)
    ax[2].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)  # Overlay GT mask with opacity
    ax[2].imshow(sam_mask, cmap=sam_cmap, alpha=0.5)  # Overlay SAM mask with opacity
    ax[2].set_title('Image with GT and SAM Masks')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def calculate_iou(gt_mask, sam_mask):
    """
    Calculate Intersection over Union (IoU) between two masks.

    Parameters:
    gt_mask (numpy.ndarray): Ground truth mask.
    sam_mask (numpy.ndarray): SAM mask.

    Returns:
    float: IoU score.
    """
    intersection = np.logical_and(gt_mask, sam_mask)
    union = np.logical_or(gt_mask, sam_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def polygon_to_mask(polygon, image_shape=(1024, 2048)):
    """
    Convert a polygon to a mask.

    Parameters:
    polygon (list): A list of tuples representing the x and y coordinates of each point in the polygon.
    image_shape (tuple): A tuple representing the shape of the image. The mask will be the same size as this shape.

    Returns:
    numpy.ndarray: A binary mask of the same size as image_shape, with the polygon filled in.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon)], 1)
    return mask

def setup_sam_model():
    sam_checkpoint = "/workspace/sam_al/model-directory/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda:0")
    predictor = SamPredictor(sam)
    return predictor, sam

def get_values_from_data_iter(data, batch_size, predictor, input_image_size=(1024, 2048)):

    # All masks from image to tensor:
    gt_mask = []
    for item in data:
        polygon = item['polygon']
        mask = polygon_to_mask(polygon)
        gt_mask.append(mask)
    mask_list = gt_mask
    gt_mask = torch.tensor(np.array(gt_mask)).to(predictor.device)

    # Create list of bboxes:
    bboxes = []
    for item in data:
        bbox = item['bbox']
        bboxes.append(bbox)
    bboxes = torch.tensor(np.array(bboxes), device=predictor.device)
    bboxes = predictor.transform.apply_boxes_torch(bboxes, input_image_size)

    # Create a list of labels:
    labels = []
    for item in data:
        label = item['label']
        labels.append(label)
    
    gt_mask_split = torch.split(gt_mask, batch_size, dim=0)
    bboxes_split = torch.split(bboxes, batch_size, dim=0)
    labels_split = [labels[i:i+batch_size] for i in range(0, len(labels), batch_size)]
    
    return gt_mask_split, bboxes_split, labels_split

def plot_and_save_masks(labels, predictions, file_name='masks_comparison.png'):
    """
    Plots and saves comparison of label and prediction masks side by side.

    Args:
    labels (torch.Tensor): A tensor of shape [4, 1024, 2048] on CUDA, representing labels.
    predictions (torch.Tensor): A tensor of shape [4, 1024, 2048] on CUDA, representing predictions.
    file_name (str): Filename to save the plot.
    """
    # Ensure the input tensors are on the same device and have the expected shape
    assert labels.shape == predictions.shape == (4, 1024, 2048), "Input tensors must have the shape [4, 1024, 2048]"
    assert labels.device == predictions.device, "Both tensors must be on the same device"

    # Move tensors to CPU and convert to NumPy arrays
    labels_np = labels.cpu().detach().numpy()
    predictions_np = predictions.cpu().detach().numpy()

    # Set up the plot with 4 rows (for each batch) and 2 columns (for labels and predictions)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 24))

    for i in range(4):
        # Plot labels
        axes[i, 0].imshow(labels_np[i], cmap='gray')
        axes[i, 0].set_title(f'Label {i}')
        axes[i, 0].axis('off')  # Turn off axis

        # Plot predictions
        axes[i, 1].imshow(predictions_np[i], cmap='gray')
        axes[i, 1].set_title(f'Prediction {i}')
        axes[i, 1].axis('off')  # Turn off axis

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# Example usage:
# Assuming 'labels' and 'predictions' are your tensors on a CUDA device
# plot_and_save_masks(labels, predictions)
torch.autograd.set_detect_anomaly(True)

def calculate_rect_size(bbox):
    # Assuming bbox is a tensor with shape [1, 4] and format [x1, y1, x2, y2]
    # Extract coordinates
    x1, y1, x2, y2 = bbox[0].cpu().numpy()
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Return size (width, height)
    return width*height

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        preds = preds.contiguous()
        labels = labels.contiguous()

        intersection = (preds * labels).sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=(1, 2)) + labels.sum(dim=(1, 2)) + self.smooth)
        
        return 1 - dice.mean()
    
def finetune_sam_model(dataset, batch_size=16, epoches=1):

    # Create a DataLoader instance for the training dataset
    from torch.utils.data import DataLoader
    from torch.nn.functional import threshold, normalize
    from torch.nn.utils import clip_grad_norm_
    from itertools import chain


    #optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-4) #  weight_decay=1e-4
    optimizer = torch.optim.Adam(chain(sam_model.mask_decoder.parameters(), sam_model.image_encoder.parameters(), sam_model.prompt_encoder.parameters()), lr=1e-5)
    loss_fn = DiceLoss(smooth=1) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    def compute_loss_and_mask(model, image_embedding, bbox, predictor_device):

        with torch.no_grad():
            # Generate prompt embeddings without gradient computation
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )

        # Generate masks using the model's mask decoder
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale masks and convert to binary format
        upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).to(predictor_device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(predictor_device)
        binary_mask = binary_mask.squeeze()
        if len(binary_mask.shape) == 2:
            binary_mask = binary_mask.unsqueeze(0)

        return binary_mask

    def get_loss(model, binary_mask, gt_mask):
        return loss_fn(binary_mask, gt_mask.float())

    for epoch in range(epoches):
        epoch_losses = []
        for index, (input_image, data) in enumerate(dataset):

            gt_mask, bboxes, labels = get_values_from_data_iter(data, batch_size, predictor)
            full_labels = labels
            
            th_size_boxes = []
            th_size_labels = []
            th_size_gt_mask = []

            for i, box in enumerate(bboxes):
                curr_size = calculate_rect_size(box)
                if curr_size > 1000 and curr_size < 130880:
                    th_size_boxes.append(box)
                    th_size_labels.append(labels[i])
                    th_size_gt_mask.append(gt_mask[i])

            bboxes =  th_size_boxes
            labels = th_size_labels
            gt_mask = th_size_gt_mask

            input_image = input_image.to(predictor.device)
            original_image_size = (1024, 2048)
            input_size = (512, 1024)            
            # with torch.no_grad():

            """ We want to embed images by wrapping the encoder in the torch.no_grad() 
            context manager, since otherwise we will have memory issues, along with 
            the fact that we are not looking to fine-tune the image encoder. """
            input_image = input_image.unsqueeze(0) # add another false dimension to input_image
            min_loss = 1000000
            k = 10  # Number of last iterations to check
            p = 5   # Minimum number of iterations with decreased loss
            loss_decreased_counter = 0  # Counter for iterations where loss decreased
            loss_history = []  # History of loss values to keep track of last k iterations
   
            for i, (curr_gt_mask, curr_bbox, curr_label) in enumerate(zip(gt_mask, bboxes, labels)):
                input_image_postprocess = sam_model.preprocess(input_image)
                image_embedding = sam_model.image_encoder(input_image_postprocess)

                binary_mask = compute_loss_and_mask(sam_model, image_embedding, curr_bbox, predictor.device)
                loss = get_loss(sam_model, binary_mask, curr_gt_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())
                # wandb.log({"Initial Loss": loss.item()})  # Log initial loss
                # Update loss history
                if len(loss_history) >= k:
                    # Remove the oldest loss if we have already k losses in history
                    oldest_loss = loss_history.pop(0)
                    # Decrease counter if the oldest loss was part of the decreased losses
                    if oldest_loss < min_loss:
                        loss_decreased_counter -= 1
                # Add current loss to history
                loss_history.append(loss.item())

                if loss.item() < min_loss:
                    min_loss = loss.item()
                    loss_decreased_counter += 1  # Increment counter as current loss is less than min_loss

                # Check if in the last k iterations at least p iterations had decreased loss
                if loss_decreased_counter >= p or i == 0:
                    visualize_and_save(input_image, curr_gt_mask.float(), binary_mask, curr_bbox, filename=f"encoder_loss_decrease_{i}.png")
                    loss_decreased_counter=0
                    v=0

                
                del loss, binary_mask, image_embedding, input_image_postprocess
                torch.cuda.empty_cache()
                i+=1

                if i%20 == 0:
                    scheduler.step()  # Update the learning rate
                    print(f"Learning Rate: {scheduler.get_lr()}")
                    
        # wandb.log({"Mean loss": np.mean(epoch_losses), "Epoch": epoch})
        # scheduler.step()  # Update the learning rate
    
    # Save the model's state dictionary to a file
    torch.save(sam_model.state_dict(), "/workspace/mask-auto-labeler/SAM_AL/fine_tune_sam_model.pth")



predictor, sam_model = setup_sam_model()
iou_dict = {}
low_flag = True
high_flag = True



# import wandb

# wandb.login()


# wandb.init(
#     # set the wandb project where this run will be logged
#     project="DAPT_CityScapes",
    
#     # track hyperparameters and run metadata
#     config={
#     "active learning method": "random",
#     "architecture": "SAM",
#     "dataset": "CityScapes",
#     "epochs": 20,
#     "batch_size": 64,
#     },
    
#     mode="disabled"
# )        

active_learning_dataset = ActiveLearningDataset(train_dataset, train_percent=0.2, sampling_method='random')
training_subset = active_learning_dataset.get_training_subset()
finetune_sam_model(dataset=training_subset, batch_size=4, epoches=10)
# wandb.finish()