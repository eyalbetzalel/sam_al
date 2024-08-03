import torch
from segment_anything import sam_model_registry, SamPredictor
from utils import get_values_from_data_iter, calculate_rect_size, visualize_and_save
import wandb
from itertools import chain
import torch.nn as nn
import numpy as np

# Dice loss implementation for segmentation tasks
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        """
        Initializes the DiceLoss module.

        Parameters:
        smooth (float): Smoothing factor to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        """
        Forward pass for Dice loss computation.

        Parameters:
        preds (torch.Tensor): Predicted masks.
        labels (torch.Tensor): Ground truth masks.

        Returns:
        torch.Tensor: Dice loss value.
        """
        preds = preds.contiguous()
        labels = labels.contiguous()

        intersection = (preds * labels).sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=(1, 2)) + labels.sum(dim=(1, 2)) + self.smooth)
        
        return 1 - dice.mean()


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1):
        """
        Initializes the JaccardLoss module.

        Parameters:
        smooth (float): Smoothing factor to avoid division by zero.
        """
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        """
        Forward pass for Jaccard loss computation.

        Parameters:
        preds (torch.Tensor): Predicted masks.
        labels (torch.Tensor): Ground truth masks.

        Returns:
        torch.Tensor: Jaccard loss value.
        """
        preds = preds.contiguous()
        labels = labels.contiguous()

        intersection = (preds * labels).sum(dim=(1, 2))
        union = preds.sum(dim=(1, 2)) + labels.sum(dim=(1, 2)) - intersection
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - jaccard.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        """
        Initializes the FocalLoss module.

        Parameters:
        alpha (float): Balancing factor.
        gamma (float): Focusing parameter.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, preds, labels):
        """
        Forward pass for Focal loss computation.

        Parameters:
        preds (torch.Tensor): Predicted masks.
        labels (torch.Tensor): Ground truth masks.

        Returns:
        torch.Tensor: Focal loss value.
        """
        logpt = -self.cross_entropy_loss(preds, labels)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        return self.alpha * focal_loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        """
        Initializes the TverskyLoss module.

        Parameters:
        alpha (float): Weight for false positives.
        beta (float): Weight for false negatives.
        smooth (float): Smoothing factor to avoid division by zero.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, labels):
        """
        Forward pass for Tversky loss computation.

        Parameters:
        preds (torch.Tensor): Predicted masks.
        labels (torch.Tensor): Ground truth masks.

        Returns:
        torch.Tensor: Tversky loss value.
        """
        preds = preds.contiguous()
        labels = labels.contiguous()

        true_pos = (preds * labels).sum(dim=(1, 2))
        false_neg = (labels * (1 - preds)).sum(dim=(1, 2))
        false_pos = ((1 - labels) * preds).sum(dim=(1, 2))
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        
        return 1 - tversky.mean()

def sam_demo_code(image, input_box, predictor):
    """
    Run the SAM model on the given image and bounding box.

    Parameters:
    image (torch.Tensor): Image tensor.
    bbox (torch.Tensor): Bounding box tensor.
    sam_model (SAM): The SAM model.
    predictor (SamPredictor): The SAM predictor.

    Returns:
    torch.Tensor: Binary mask.
    """
    # input_box = np.array([600, 1100, 2100, 1800])
    # image_path = '/kaggle/input/happy-whale-and-dolphin/train_images/00177f3c614d1e.jpg'
    # image_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    return masks[0]

def setup_sam_model():
    """
    Setup the SAM model and predictor.

    Returns:
    SamPredictor: SAM predictor.
    SAM: SAM model.
    """
    sam_checkpoint = "/workspace/sam_al/model-directory/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda:0")
    predictor = SamPredictor(sam)
    return predictor, sam

def finetune_sam_model(sam_model, predictor, train_dataset, validation_dataset, batch_size=4, epoches=1, patience=3, iter_num=0, lr=1e-5):
    """
    Fine-tune the SAM model on the given dataset.

    Parameters:
    sam_model (SAM): The SAM model.
    predictor (SamPredictor): The SAM predictor.
    train_dataset (Dataset): The training dataset.
    validation_dataset (Dataset): The validation dataset.
    batch_size (int): Batch size for training.
    epoches (int): Number of epochs to train.
    patience (int): Number of epochs with no improvement after which training will be stopped.
    iter_num (int): Current iteration number.
    lr (float): Learning rate.

    Returns:
    None
    """
    # Setup optimizer, loss function, and scheduler
    # optimizer = torch.optim.Adam(chain(sam_model.mask_decoder.parameters(), sam_model.image_encoder.parameters(), sam_model.prompt_encoder.parameters()), lr=lr)
    # optimizer = torch.optim.Adam(chain(sam_model.mask_decoder.parameters(), sam_model.image_encoder.parameters()), lr=lr)
    optimizer = torch.optim.Adam(chain(sam_model.mask_decoder.parameters()), lr=lr)
    loss_fn = DiceLoss(smooth=1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    def clip_mask_to_bbox(mask, bbox, delta=0.1):
        
        """
        Clip the mask to the bounding box with an allowed extension of delta percentage.

        Parameters:
        mask (torch.Tensor): Predicted mask.
        bbox (torch.Tensor): Bounding box tensor.
        delta (float): Percentage by which the mask is allowed to extend beyond the bounding box.

        Returns:
        torch.Tensor: Clipped mask.
        """
        height, width = mask.shape[-2:]
        bbox = bbox.cpu().numpy().astype(int).tolist()[0]
        # Calculate the extended bounding box dimensions
        x_min, y_min, x_max, y_max = 2*bbox[0], 2*bbox[1], 2*bbox[2], 2*bbox[3]
        box_width = x_max - x_min
        box_height = y_max - y_min

        x_min_extended = max(0, x_min - delta * box_width)
        y_min_extended = max(0, y_min - delta * box_height)
        x_max_extended = min(width, x_max + delta * box_width)
        y_max_extended = min(height, y_max + delta * box_height)

        # Create the clipped mask
        clipped_mask = torch.zeros_like(mask)
        clipped_mask[:, int(y_min_extended):int(y_max_extended), int(x_min_extended):int(x_max_extended)] = mask[:, int(y_min_extended):int(y_max_extended), int(x_min_extended):int(x_max_extended)]
        
        return clipped_mask

    def compute_loss_and_mask(model, image_embedding, bbox, predictor_device, input_size, original_image_size, delta=0.1):
        """
        Compute the loss and mask for a given image embedding and bounding box.

        Parameters:
        model (SAM): The SAM model.
        image_embedding (torch.Tensor): Image embedding tensor.
        bbox (torch.Tensor): Bounding box tensor.
        predictor_device (str): Device to perform computation on.
        delta (float): Percentage by which the mask is allowed to extend beyond the bounding box.

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

        # Clip the mask to the bounding box with an extension delta
        binary_mask = clip_mask_to_bbox(binary_mask, bbox, delta)

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

    def validate_model(validation_dataset):
        """
        Validate the model on the validation dataset.

        Parameters:
        validation_dataset (Dataset): The validation dataset.

        Returns:
        float: The average validation loss.
        """
        sam_model.eval()
        validation_loss = []
        input_size = (512, 1024)
        original_image_size = (1024, 2048)
        with torch.no_grad():
            for input_image, data in validation_dataset:
                if data is None:
                    continue
                gt_mask, bboxes, labels = get_values_from_data_iter(data, batch_size, predictor)
                if len(labels) == 0:
                    continue
                input_image = input_image.to(predictor.device)
                input_image = input_image.unsqueeze(0)
                for curr_gt_mask, curr_bbox in zip(gt_mask, bboxes):
                    input_image_postprocess = sam_model.preprocess(input_image)
                    image_embedding = sam_model.image_encoder(input_image_postprocess)
                    binary_mask = compute_loss_and_mask(sam_model, image_embedding, curr_bbox, predictor.device, input_size, original_image_size)
                    loss = get_loss(sam_model, binary_mask, curr_gt_mask)
                    validation_loss.append(loss.item())
        sam_model.train()
        return np.mean(validation_loss)
    
    best_val_loss = float('inf')
    # best_val_loss = 100000
    epochs_without_improvement = 0
    

    for epoch in range(epoches):
        singleImageLoggingFlag = True
        for index, (input_image, data) in enumerate(train_dataset):
            (input_image, data) = train_dataset[0]

            ########## Demo Sanity Check - 1 ##########

            input_demo = input_image
            predictor_demo, _ = setup_sam_model()

            ###########################################

            if data is None:
                continue
            gt_mask, bboxes, labels = get_values_from_data_iter(data, batch_size, predictor)
            if len(labels) == 0:
                continue

            input_image = input_image.to(predictor.device)
            original_image_size = (1024, 2048)
            input_size = (512, 1024)
            input_image = input_image.unsqueeze(0)
   
            for i, (curr_gt_mask, curr_bbox, curr_label) in enumerate(zip(gt_mask, bboxes, labels)):
                
                ########## Demo Sanity Check - 2 ##########
                v=0
                mask_demo = sam_demo_code(input_demo, curr_bbox, predictor_demo)
                v=0
                ###########################################
                
                input_image_postprocess = sam_model.preprocess(input_image)
                with torch.no_grad():
                    image_embedding = sam_model.image_encoder(input_image_postprocess)

                binary_mask = compute_loss_and_mask(sam_model, image_embedding, curr_bbox, predictor.device, input_size, original_image_size)
                loss = get_loss(sam_model, binary_mask, curr_gt_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({f"Iteration_{iter_num + 1}/Epoch": epoch, f"Iteration_{iter_num + 1}/Train Loss": loss.item()})
                
                if singleImageLoggingFlag:
                    visualize_and_save(input_image, curr_gt_mask, binary_mask, curr_bbox, filename="test_sam.png")
                    wandb.log({f"Iteration_{iter_num + 1}/Validation Image | epoch {epoch}": [wandb.Image("test_sam.png")]})
                    singleImageLoggingFlag = False

                del loss, binary_mask, image_embedding, input_image_postprocess
                torch.cuda.empty_cache()


                
        val_loss = validate_model(validation_dataset)

        wandb.log({f"Iteration_{iter_num + 1}/Epoch": epoch, 
                   f"Iteration_{iter_num + 1}/Validation Loss": val_loss, 
                   f"Iteration_{iter_num + 1}/lr": lr})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

    # Save model to disk
    # torch.save(sam_model.state_dict(), "/workspace/mask-auto-labeler/SAM_AL/fine_tune_sam_model.pth")
