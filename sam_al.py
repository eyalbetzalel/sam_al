
# TODO 6 : Evaluate the model's performance unsupervisedly on the data
# TODO 7 : Implement active learning strategy (e.g., uncertainty sampling, query-by-committee, etc.)
# TODO 9 : Annotate the selected instances unsupervisedly
# TODO 10: Retrain the model with the newly annotated instances (fine-tuning SAM)
# TODO 11: Repeat steps 6-10 until satisfactory performance is achieved

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

def visualize_and_save(image, gt_mask, sam_mask, filename):
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
    # Create custom colormaps
    gt_cmap = ListedColormap(['none', 'red'])
    sam_cmap = ListedColormap(['none', 'blue'])

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image)
    ax[0].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)  # Overlay GT mask with opacity
    ax[0].set_title('Image with GT Mask')

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
    sam_checkpoint = "/workspace/mask-auto-labeler/SAM_AL/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda:1")
    predictor = SamPredictor(sam)
    return predictor, sam

def finetune_sam_model(dataset, batch_size=4, epoches=10):

    # Create a DataLoader instance for the training dataset
    from torch.utils.data import DataLoader
    from torch.nn.functional import threshold, normalize


    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters())
    loss_fn = torch.nn.MSELoss()

    for input_image, data in dataset:
        # data is a list of dictionaries with keys 'label', 'polygon', and 'bbox'. this code do the following:
        # 1. Convert the polygon to a mask and put it in a tensors (gt_mask)
        # 2. Create list of bboxes
        # 3. Create a list of labels

        # Convert the polygon to a mask and put it in a tensors (gt_mask):
        input_image = input_image.to(predictor.device)
        
        original_image_size = (1024, 2048)
        input_size = (1024, 2048)
        gt_mask = []
        for item in data:
            polygon = item['polygon']
            mask = polygon_to_mask(polygon)
            gt_mask.append(mask)
        gt_mask = torch.tensor(gt_mask).to(predictor.device)

        # Create list of bboxes:
        bboxes = []
        for item in data:
            bbox = item['bbox']
            bboxes.append(bbox)
        bboxes = torch.tensor(bboxes, device=predictor.device)

        # Create a list of labels:
        labels = []
        for item in data:
            label = item['label']
            labels.append(label)
        
        # Set the image to the predictor

        transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, input_image.shape[:2])
        
        with torch.no_grad():

            """ We want to embed images by wrapping the encoder in the torch.no_grad() 
            context manager, since otherwise we will have memory issues, along with 
            the fact that we are not looking to fine-tune the image encoder. """
            # add another false dimension to input_image
            input_image = input_image.unsqueeze(0)
            input_image_postprocess = sam_model.preprocess(input_image)
            image_embedding = sam_model.image_encoder(input_image_postprocess)
        
        with torch.no_grad():
            
            """ We can also generate the prompt embeddings within the no_grad context manager. 
            We use our bounding box coordinates, converted to pytorch tensors.
            """
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=transformed_boxes[:4,:],
                masks=None,
            )
        """ Finally, we can generate the masks. Note that here we are in single mask generation
            mode (in contrast to the 3 masks that are normally output).""" 
        
        low_res_masks, iou_predictions, upscaled_embedding = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        )

        """ The final step here is to upscale the masks back to the original image size since they are low resolution.
         We can use Sam.postprocess_masks to achieve this. We will also want to generate binary masks from the 
         predicted masks so that we can compare these to our ground truths. It is important to use torch functionals 
         in order to not break backpropagation."""
    
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(predictor.device)

        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(predictor.device)
        binary_mask = binary_mask.squeeze()
        gt_mask = gt_mask[:4,:]

        loss = loss_fn(binary_mask, gt_mask.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

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
        
active_learning_dataset = ActiveLearningDataset(train_dataset, 0.1)
training_subset = active_learning_dataset.get_training_subset()
finetune_sam_model(dataset=training_subset, batch_size=4, epoches=10)
v=0