

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

def get_values_from_data_iter(data, batch_size, predictor, input_image_size=(1024, 2048)):

    # All masks from image to tensor:
    gt_mask = []
    for item in data:
        polygon = item['polygon']
        mask = polygon_to_mask(polygon)
        gt_mask.append(mask)
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

def finetune_sam_model(dataset, batch_size=16, epoches=1):

    # Create a DataLoader instance for the training dataset
    from torch.utils.data import DataLoader
    from torch.nn.functional import threshold, normalize

    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    loss_fn = torch.nn.MSELoss() # TODO : #Try DiceFocalLoss, FocalLoss, DiceCELoss

    for epoch in range(epoches):
        epoch_losses = []
        for index, (input_image, data) in enumerate(dataset):

            input_image = input_image.to(predictor.device)
            original_image_size = (1024, 2048)
            input_size = (1024, 2048)            
            with torch.no_grad():

                """ We want to embed images by wrapping the encoder in the torch.no_grad() 
                context manager, since otherwise we will have memory issues, along with 
                the fact that we are not looking to fine-tune the image encoder. """
                input_image = input_image.unsqueeze(0) # add another false dimension to input_image
                input_image_postprocess = sam_model.preprocess(input_image)
                image_embedding = sam_model.image_encoder(input_image_postprocess)
            
            gt_mask, bboxes, labels = get_values_from_data_iter(data, batch_size, predictor)
                        
            for i, (curr_gt_mask, curr_bbox, curr_label) in enumerate(zip(gt_mask, bboxes, labels)):

                with torch.no_grad():
                    
                    """ We can also generate the prompt embeddings within the no_grad context manager. 
                    We use our bounding box coordinates, converted to pytorch tensors.
                    """
                    
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None,
                        boxes=curr_bbox,
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
                if len(binary_mask.shape) == 2:
                    binary_mask = binary_mask.unsqueeze(0)

                loss = loss_fn(binary_mask, curr_gt_mask.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

        print(f"Mean loss: {np.mean(epoch_losses)} | Epoch: {epoch}")
    
    # Save the model's state dictionary to a file
    torch.save(sam_model.state_dict(), "/workspace/mask-auto-labeler/SAM_AL/fine_tune_sam_model.pth")

# TODO, Finish this function:
def evaluate_iou_per_class(model, dataset, device, batch_size=4):


    # Dictionary to hold IoU sums and count per class
    class_iou = defaultdict(lambda: {'iou_sum': 0.0, 'count': 0})
    
    # Model in evaluation mode
    model.eval()

    with torch.no_grad():  # No gradients needed
        for index, (input_image, data) in enumerate(dataset):
            input_image = input_image.to(predictor.device)  # Assume 'device' is defined
            input_image_postprocess = model.preprocess(input_image)
            image_embedding = model.image_encoder(input_image_postprocess)

            gt_mask, bboxes, labels = get_values_from_data_iter(data, batch_size, predictor)  # Assuming this function is defined
            
            for curr_gt_mask, curr_bbox, curr_label in zip(gt_mask, bboxes, labels):
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=curr_bbox,
                    masks=None,
                )
                
                low_res_masks, _, _ = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Upscale and threshold masks
                original_image_size = (1024, 2048)
                upscaled_masks = model.postprocess_masks(low_res_masks, input_image.shape[-2:], original_image_size)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(predictor.device)
                binary_mask = binary_mask.squeeze()

                # Calculate IoU for each class
                for i, label in enumerate(curr_label):
                    intersection = (binary_mask[i] * curr_gt_mask[i]).sum()
                    union = (binary_mask[i] + curr_gt_mask[i] - (binary_mask[i] * curr_gt_mask[i])).sum()
                    iou = intersection / union if union > 0 else 0

                    class_iou[label.item()]['iou_sum'] += iou.item()
                    class_iou[label.item()]['count'] += 1

    # Prepare the results in a DataFrame
    results = {'Class': [], 'Mean IoU': [], 'Std IoU': []}
    for label, metrics in class_iou.items():
        mean_iou = metrics['iou_sum'] / metrics['count']
        # Collect IoUs to calculate standard deviation
        ious = [((binary_mask[i] * gt_mask[i]).sum() / ((binary_mask[i] + gt_mask[i] - (binary_mask[i] * gt_mask[i])).sum())).item()
                for i, l in enumerate(labels) if l.item() == label]
        std_iou = torch.std(torch.tensor(ious)).item()

        results['Class'].append(label)
        results['Mean IoU'].append(mean_iou)
        results['Std IoU'].append(std_iou)

    return pd.DataFrame(results)

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
finetune_sam_model(dataset=training_subset, batch_size=64, epoches=20)

# TODO 1 : Save model after iteration of traning on random subset of data [V]
# TODO 2 : Evaluate the model's performance on the validation data
# TODO 3 : Implement active learning strategy 
# TODO 4 : Evaluate the model's performance unsupervisedly on the data
# TODO 5 : Annotate the selected instances unsupervisedly (????????)
# TODO 6: Retrain the model with the newly annotated instances (fine-tuning SAM)
# TODO 7: Repeat steps 6-10 until satisfactory performance is achieved
v=0