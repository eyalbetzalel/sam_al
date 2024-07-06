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
import random
from itertools import chain

# Dice loss implementation for segmentation tasks
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

def visualize_and_save(image, gt_mask, sam_mask, curr_bbox, filename="test_sam.png"):
    """
    Visualize and save the image, ground truth mask, and SAM mask.

    Parameters:
    image (numpy.ndarray): The original image.
    gt_mask (numpy.ndarray): Ground truth mask.
    sam_mask (numpy.ndarray): SAM mask.
    curr_bbox (numpy.ndarray): Bounding box.
    filename (str): The filename to save the image to.

    Returns:
    None
    """
    gt_mask = gt_mask.detach().cpu().numpy()[0,:,:]
    sam_mask = sam_mask.detach().cpu().numpy()[0,:,:]
    curr_bbox = curr_bbox.detach().cpu().numpy()[0,:]
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
    """
    Setup the SAM model and predictor.

    Returns:
    predictor (SamPredictor): SAM predictor.
    sam (SAM): SAM model.
    """
    sam_checkpoint = "/workspace/sam_al/model-directory/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda:0")
    predictor = SamPredictor(sam)
    return predictor, sam

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
    mask_list = gt_mask
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

def plot_and_save_masks(labels, predictions, file_name='masks_comparison.png'):
    """
    Plots and saves comparison of label and prediction masks side by side.

    Args:
    labels (torch.Tensor): A tensor of shape [4, 1024, 2048] on CUDA, representing labels.
    predictions (torch.Tensor): A tensor of shape [4, 1024, 2048] on CUDA, representing predictions.
    file_name (str): Filename to save the plot.
    """
    assert labels.shape == predictions.shape == (4, 1024, 2048), "Input tensors must have the shape [4, 1024, 2048]"
    assert labels.device == predictions.device, "Both tensors must be on the same device"

    labels_np = labels.cpu().detach().numpy()
    predictions_np = predictions.cpu().detach().numpy()

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 24))

    for i in range(4):
        axes[i, 0].imshow(labels_np[i], cmap='gray')
        axes[i, 0].set_title(f'Label {i}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(predictions_np[i], cmap='gray')
        axes[i, 1].set_title(f'Prediction {i}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

torch.autograd.set_detect_anomaly(True)

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

def finetune_sam_model(dataset, batch_size=4, epoches=1):
    """
    Fine-tune the SAM model on the given dataset.

    Parameters:
    dataset (Dataset): The training dataset.
    batch_size (int): Batch size for training.
    epoches (int): Number of epochs to train.

    Returns:
    None
    """
    optimizer = torch.optim.Adam(chain(sam_model.mask_decoder.parameters(), sam_model.image_encoder.parameters(), sam_model.prompt_encoder.parameters()), lr=1e-5)
    loss_fn = DiceLoss(smooth=1) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    def compute_loss_and_mask(model, image_embedding, bbox, predictor_device):
        """
        Compute the loss and mask for a given image embedding and bounding box.

        Parameters:
        model (SAM): The SAM model.
        image_embedding (torch.Tensor): Image embedding tensor.
        bbox (torch.Tensor): Bounding box tensor.
        predictor_device (str): Device to perform computation on.

        Returns:
        torch.Tensor: Binary mask.
        """
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=bbox, masks=None)

        low_res_masks, iou_predictions = model.mask_decoder(image_embeddings=image_embedding, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)

        upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).to(predictor_device)
        binary_mask = torch.nn.functional.normalize(torch.nn.functional.threshold(upscaled_masks, 0.0, 0)).to(predictor_device)
        binary_mask = binary_mask.squeeze()
        if len(binary_mask.shape) == 2:
            binary_mask = binary_mask.unsqueeze(0)

        return binary_mask

    def get_loss(model, binary_mask, gt_mask):
        """
        Compute the loss between the binary mask and ground truth mask.

        Parameters:
        model (SAM): The SAM model.
        binary_mask (torch.Tensor): Binary mask tensor.
        gt_mask (torch.Tensor): Ground truth mask tensor.

        Returns:
        torch.Tensor: Loss value.
        """
        return loss_fn(binary_mask, gt_mask.float())

    for epoch in range(epoches):
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
            input_image = input_image.unsqueeze(0)
            min_loss = 1000000
            k = 10
            p = 5
            loss_decreased_counter = 0
            loss_history = []
   
            for i, (curr_gt_mask, curr_bbox, curr_label) in enumerate(zip(gt_mask, bboxes, labels)):
                input_image_postprocess = sam_model.preprocess(input_image)
                image_embedding = sam_model.image_encoder(input_image_postprocess)

                binary_mask = compute_loss_and_mask(sam_model, image_embedding, curr_bbox, predictor.device)
                loss = get_loss(sam_model, binary_mask, curr_gt_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())

                if len(loss_history) >= k:
                    oldest_loss = loss_history.pop(0)
                    if oldest_loss < min_loss:
                        loss_decreased_counter -= 1
                loss_history.append(loss.item())

                # if loss.item() < min_loss:
                #     min_loss = loss.item()
                #     loss_decreased_counter += 1

                # if loss_decreased_counter >= p or i == 0:
                #     visualize_and_save(input_image, curr_gt_mask.float(), binary_mask, curr_bbox, filename=f"encoder_loss_decrease_{i}.png")
                #     loss_decreased_counter = 0
                
                del loss, binary_mask, image_embedding, input_image_postprocess
                torch.cuda.empty_cache()
                i += 1

                # if i % 20 == 0:
                #     scheduler.step()
                #     print(f"Learning Rate: {scheduler.get_lr()}")
                    
    # torch.save(sam_model.state_dict(), "/workspace/mask-auto-labeler/SAM_AL/fine_tune_sam_model.pth")

predictor, sam_model = setup_sam_model()

def random_query_strategy(unlabeled_subset, num_samples):
    """Randomly select samples from the unlabeled dataset."""
    return random.sample(range(len(unlabeled_subset)), num_samples)

# Addon: ActiveLearningPlatform class
class ActiveLearningPlatform:
    def __init__(self, model, predictor, initial_dataset, batch_size, max_iterations, query_strategy):
        """
        Initialize the Active Learning Platform.

        Parameters:
        model (SAM): The SAM model.
        predictor (SamPredictor): The SAM predictor.
        initial_dataset (Dataset): The initial dataset.
        batch_size (int): Batch size for training.
        max_iterations (int): Maximum number of active learning iterations.
        query_strategy (function): Function to select samples from the unlabeled dataset.
        """
        self.model = model
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.query_strategy = query_strategy  # Addon: query strategy input
        self.active_learning_dataset = ActiveLearningDataset(initial_dataset, train_percent=0.001, sampling_method='random')

    def train_model(self):
        """Train the model on the current labeled dataset."""
        training_subset = self.active_learning_dataset.get_training_subset()
        finetune_sam_model(training_subset, batch_size=self.batch_size, epoches=1)

    def perform_inference(self):
        """Perform inference on the unlabeled dataset (for advanced strategies)."""
        pass

    def query_labels(self):
        """
        Query labels from the unlabeled dataset using the query strategy.

        Returns:
        list: Indices of selected samples from the unlabeled dataset.
        """
        unlabeled_subset = self.active_learning_dataset.get_unlabeled_subset()
        num_samples_to_query = min(len(unlabeled_subset), self.batch_size) # TODO : Fix this part with an hyperparameter. the batch size isn't for the amount of images is for the amount of masks.
        queried_indices = self.query_strategy(unlabeled_subset, num_samples_to_query)  # Addon: use query strategy
        return queried_indices

    def update_datasets(self, new_indices):
        """Update the datasets by moving newly labeled samples from the unlabeled to labeled set."""
        self.active_learning_dataset.update_labeled_set(new_indices)

    def run(self):
        """Run the active learning loop."""
        for iteration in range(self.max_iterations):
            self.train_model()
            queried_indices = self.query_labels()
            self.update_datasets(queried_indices)
            print(f"Iteration {iteration + 1}/{self.max_iterations} complete")
        print("Active learning process complete")

# Example Usage
initial_dataset = train_dataset
batch_size = 4
max_iterations = 10
query_strategy = random_query_strategy  # Addon: define query strategy

# Addon: initialize and run the active learning platform
active_learning_platform = ActiveLearningPlatform(sam_model, predictor, initial_dataset, batch_size, max_iterations, query_strategy)
active_learning_platform.run()

# TODO 1: Implement a more advanced query strategy for active learning.
# TODO 2: Implement a method to evaluate the model on the validation set.
# TODO 3: Implement a method to visualize the model predictions on the test set.
# TODO 4: Implement a method to save the trained model to disk.
# TODO 5: Implement a method to load a trained model from disk.
# TODO 6: 