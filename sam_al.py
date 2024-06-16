

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
    sam_checkpoint = "/workspace/sam_al/model-directory/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda:1")
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

def finetune_sam_model(dataset, batch_size=16, epoches=1):

    

    # Create a DataLoader instance for the training dataset
    from torch.utils.data import DataLoader
    from torch.nn.functional import threshold, normalize
    from torch.nn.utils import clip_grad_norm_
    from itertools import chain


    #optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-4) #  weight_decay=1e-4
    optimizer = torch.optim.Adam(chain(sam_model.mask_decoder.parameters(), sam_model.image_encoder.parameters(), sam_model.prompt_encoder.parameters()), lr=1e-5)
    loss_fn = torch.nn.MSELoss() # TODO : #Try DiceFocalLoss, FocalLoss, DiceCELoss
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

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

            input_image = input_image.to(predictor.device)
            original_image_size = (1024, 2048)
            input_size = (512, 1024)            
            with torch.no_grad():

                """ We want to embed images by wrapping the encoder in the torch.no_grad() 
                context manager, since otherwise we will have memory issues, along with 
                the fact that we are not looking to fine-tune the image encoder. """
                input_image = input_image.unsqueeze(0) # add another false dimension to input_image
                input_image_postprocess = sam_model.preprocess(input_image)
                image_embedding = sam_model.image_encoder(input_image_postprocess)
            
            gt_mask, bboxes, labels = get_values_from_data_iter(data, batch_size, predictor)
            
            if len(dataset) == 1:
                gt_mask = [gt_mask[36]]
                bboxes = [bboxes[36]]
                labels = [labels[36]]

            max_norms = []            
            
            for i, (curr_gt_mask, curr_bbox, curr_label) in enumerate(zip(gt_mask, bboxes, labels)):

                binary_mask = compute_loss_and_mask(sam_model, image_embedding, curr_bbox, predictor.device)
                loss = get_loss(sam_model, binary_mask, curr_gt_mask)
                wandb.log({"Initial Loss": loss.item()})  # Log initial loss

                # If the loss is high, train on this image until the loss is reduced
                if loss > 0 :
                    visualize_and_save(input_image, curr_gt_mask.float(), binary_mask, curr_bbox, filename=f"big_loss_start_{index}.png")
                    while loss > 0:
                        print(loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        # clip_grad_norm_(sam_model.mask_decoder.parameters(), max_norm=1e-4)
                        optimizer.step()

                        # Recompute loss and binary mask after updating
                        binary_mask = compute_loss_and_mask(sam_model, image_embedding, curr_bbox, predictor.device)
                        loss = get_loss(sam_model, binary_mask, curr_gt_mask)  # Update the loss variable
                        wandb.log({"Updated Loss": loss.item()})  # Log updated loss
                        
                    # Optionally, save or visualize the current mask
                else:
                    continue

                visualize_and_save(input_image, curr_gt_mask.float(), binary_mask, curr_bbox, filename=f"big_loss_end_{index}.png")       

        wandb.log({"Mean loss": np.mean(epoch_losses), "Epoch": epoch})
        # scheduler.step()  # Update the learning rate
    
    # Save the model's state dictionary to a file
    torch.save(sam_model.state_dict(), "/workspace/mask-auto-labeler/SAM_AL/fine_tune_sam_model.pth")

# # TODO, Finish this function:
# def evaluate_iou_per_class(model, dataset, device, batch_size=4):


#     # Dictionary to hold IoU sums and count per class
#     class_iou = defaultdict(lambda: {'iou_sum': 0.0, 'count': 0})
    
#     # Model in evaluation mode
#     model.eval()

#     with torch.no_grad():  # No gradients needed
#         for index, (input_image, data) in enumerate(dataset):
#             input_image = input_image.to(predictor.device)  # Assume 'device' is defined
#             input_image_postprocess = model.preprocess(input_image)
#             image_embedding = model.image_encoder(input_image_postprocess)

#             gt_mask, bboxes, labels = get_values_from_data_iter(data, batch_size, predictor)  # Assuming this function is defined
            
#             for curr_gt_mask, curr_bbox, curr_label in zip(gt_mask, bboxes, labels):
#                 sparse_embeddings, dense_embeddings = model.prompt_encoder(
#                     points=None,
#                     boxes=curr_bbox,
#                     masks=None,
#                 )
                
#                 low_res_masks, _, _ = model.mask_decoder(
#                     image_embeddings=image_embedding,
#                     image_pe=model.prompt_encoder.get_dense_pe(),
#                     sparse_prompt_embeddings=sparse_embeddings,
#                     dense_prompt_embeddings=dense_embeddings,
#                     multimask_output=False,
#                 )

#                 # Upscale and threshold masks
#                 original_image_size = (1024, 2048)
#                 upscaled_masks = model.postprocess_masks(low_res_masks, input_image.shape[-2:], original_image_size)
#                 binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(predictor.device)
#                 binary_mask = binary_mask.squeeze()

#                 # Calculate IoU for each class
#                 for i, label in enumerate(curr_label):
#                     intersection = (binary_mask[i] * curr_gt_mask[i]).sum()
#                     union = (binary_mask[i] + curr_gt_mask[i] - (binary_mask[i] * curr_gt_mask[i])).sum()
#                     iou = intersection / union if union > 0 else 0

#                     class_iou[label.item()]['iou_sum'] += iou.item()
#                     class_iou[label.item()]['count'] += 1

#     # Prepare the results in a DataFrame
#     results = {'Class': [], 'Mean IoU': [], 'Std IoU': []}
#     for label, metrics in class_iou.items():
#         mean_iou = metrics['iou_sum'] / metrics['count']
#         # Collect IoUs to calculate standard deviation
#         ious = [((binary_mask[i] * gt_mask[i]).sum() / ((binary_mask[i] + gt_mask[i] - (binary_mask[i] * gt_mask[i])).sum())).item()
#                 for i, l in enumerate(labels) if l.item() == label]
#         std_iou = torch.std(torch.tensor(ious)).item()

#         results['Class'].append(label)
#         results['Mean IoU'].append(mean_iou)
#         results['Std IoU'].append(std_iou)

#     return pd.DataFrame(results)

predictor, sam_model = setup_sam_model()
iou_dict = {}
low_flag = True
high_flag = True

# for x in test_dataset:
#     img, (inst, col, poly) = x
#     cv_image = np.array(img) # Convert PIL to OpenCV
#     cv_image = cv_image[:, :, ::-1].copy() # Convert RGB to BGR 
#     predictor.set_image(cv_image) # Set the image to the predictor
#     for item in poly['objects']:
#         polygon = item['polygon']
#         label = item['label']
#         bbox = np.array(cv2.boundingRect(np.array(polygon))) # Get GT bbox of item
#         sam_mask, _, _, _ = predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             box=bbox[None, :],
#             multimask_output=False,) # Get SAM mask of item
#         sam_mask = sam_mask.squeeze()
#         gt_mask = polygon_to_mask(polygon)
#         iou = calculate_iou(gt_mask, sam_mask)
#         if label not in iou_dict:
#             iou_dict[label] = [iou]
#         else:
#             iou_dict[label].append(iou)   
#         if iou < 0.2 and low_flag:
#             visualize_and_save(cv_image, gt_mask, sam_mask, f'output_{label}_low_iou_{iou}.png')
#             low_flag = False
#         if iou > 0.8 and high_flag:
#             visualize_and_save(cv_image, gt_mask, sam_mask, f'output_{label}_high_iou_{iou}.png')
#             high_flag = False

import wandb

wandb.login()


wandb.init(
    # set the wandb project where this run will be logged
    project="DAPT_CityScapes",
    
    # track hyperparameters and run metadata
    config={
    "active learning method": "random",
    "architecture": "SAM",
    "dataset": "CityScapes",
    "epochs": 20,
    "batch_size": 64,
    },
    
    mode="disabled"
)        

active_learning_dataset = ActiveLearningDataset(train_dataset, train_percent=0.2, sampling_method='fixed')
training_subset = active_learning_dataset.get_training_subset()
finetune_sam_model(dataset=training_subset, batch_size=1, epoches=10)
wandb.finish()

# Debug : 

# Draw with images, bbox, gt_mask, sam_mask and compare
# _____

# TODO 1 : Save model after iteration of traning on random subset of data [V]
# TODO 2 : Evaluate the model's performance on the validation data
# TODO 3 : Implement active learning strategy 
# TODO 4 : Evaluate the model's performance unsupervisedly on the data
# TODO 5 : Annotate the selected instances unsupervisedly (????????)
# TODO 6: Retrain the model with the newly annotated instances (fine-tuning SAM)
# TODO 7: Repeat steps 6-10 until satisfactory performance is achieved
v=0
# what is weight decay? how it works? which value to put as default and how?