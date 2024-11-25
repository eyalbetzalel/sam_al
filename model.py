import torch
from segment_anything import sam_model_registry, SamPredictor
from utils import get_values_from_data_iter, calculate_rect_size, visualize_and_save
import wandb
from itertools import chain
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import WarmupScheduler
from torch.nn.utils import clip_grad_norm_

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
    def convert_tensor_to_image(tensor: torch.Tensor, image_format: str = "RGB") -> np.ndarray:
        """
        Converts a torch tensor with values in the range [0, 1] to a NumPy array
        in the format expected by the set_image function.

        Arguments:
            tensor (torch.Tensor): The input image tensor in CHW format with pixel values in [0, 1].
            image_format (str): The color format of the image, in ['RGB', 'BGR'].

        Returns:
            np.ndarray: The converted image in HWC uint8 format, with pixel values in [0, 255].
        """
        # Check if tensor is in CHW format
        if tensor.ndimension() != 3 or tensor.size(0) not in {1, 3}:
            raise ValueError("Input tensor must be in CHW format with 1 or 3 channels.")

        # Convert tensor to numpy array
        image_np = tensor.cpu().numpy()

        # Scale values to [0, 255] and convert to uint8
        image_np = (image_np * 255).astype(np.uint8)

        # Convert from CHW to HWC format
        image_np = np.transpose(image_np, (1, 2, 0))

        # Handle grayscale images (1 channel)
        if image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)

        if image_format == "BGR":
            image_np = image_np[..., ::-1]

        return image_np
    
    def boolean_array_to_tensor(boolean_array: np.ndarray) -> torch.Tensor:
        """
        Converts a boolean NumPy array to a PyTorch tensor with values 0 and 1.

        Arguments:
            boolean_array (np.ndarray): The input boolean array.

        Returns:
            torch.Tensor: The converted PyTorch tensor with values 0 and 1.
        """
        if boolean_array.dtype != bool:
            raise ValueError("Input array must be of boolean type.")
        
        # Convert boolean array to integer array (0 and 1)
        int_array = boolean_array.astype(np.int32)

        # Convert integer array to PyTorch tensor
        tensor = torch.from_numpy(int_array)

        return tensor

    image = convert_tensor_to_image(image)
    predictor.set_image(image)
    input_box = input_box.cpu().numpy().astype(int)[0]
    input_box = input_box * 2
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    mask_demo = boolean_array_to_tensor(masks[0])

    return mask_demo

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

def finetune_sam_model(
    sam_model, predictor, train_dataset, validation_dataset, batch_size=4, epoches=1, patience=3, 
    iter_num=0, lr=1e-5, warmup_steps=5, max_grad_norm=1.0, optimizer_name='Adam', gamma=0.8, 
    step_size=5
):
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
    warmup_steps (int): Number of warmup steps.
    max_grad_norm (float): Maximum gradient norm for clipping.
    optimizer_name (str): Name of the optimizer to use ('Adam' or 'SGD').
    gamma (float): Multiplicative factor of learning rate decay.
    step_size (int): Period of learning rate decay.

    Returns:
    float: The best validation loss.
    """
    # Setup optimizer based on optimizer_name
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(
            chain(sam_model.mask_decoder.parameters(), sam_model.image_encoder.parameters()),
            lr=lr, momentum=0.9
        )
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            chain(sam_model.mask_decoder.parameters(), sam_model.image_encoder.parameters()),
            lr=lr
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    loss_fn = DiceLoss(smooth=1)

    # Initialize the warmup scheduler
    initial_lr = lr * 0.01  # Starting small, typically 10% of the target lr
    warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps, initial_lr=initial_lr, final_lr=lr)

    # Setup learning rate scheduler with gamma and step_size
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
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
        # binary_mask = clip_mask_to_bbox(binary_mask, bbox, delta)

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

    def calculate_iou(mask1, mask2):
        # Move masks from CUDA to CPU
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        
        # Ensure masks are binary
        mask1 = (mask1 > 0).float()
        mask2 = (mask2 > 0).float()
        
        # Calculate intersection and union
        intersection = torch.sum(mask1 * mask2)
        union = torch.sum(mask1) + torch.sum(mask2) - intersection
        
        # Compute IoU
        iou = intersection / union
        
        return iou.item()

    def validate_model(validation_dataset, epoch, iter_num):
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
        class_losses = {}
        class_ious = {}
        with torch.no_grad():
            for input_image, data in validation_dataset:
                if data is None:
                    continue
                gt_mask, bboxes, labels = get_values_from_data_iter(data, batch_size, predictor)
                
                if labels == None:
                    continue
                if len(labels) == 0:
                    continue

                input_image = input_image.to(predictor.device)
                original_image_size = (1024, 2048)
                input_size = (512, 1024)

                input_image = input_image.permute(1, 2 , 0)
                input_image = input_image.cpu().numpy()
                input_image = input_image * 255
                input_image = input_image.astype(np.uint8)
                input_image = predictor.transform_image(input_image) # change shape 2048 --> 1024
                input_image_torch = torch.as_tensor(input_image, device=predictor.device) # uint8 0:255 values - torch.Size([512, 1024, 3]) 
                input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :] # uint8 0:255 values - torch.Size([1, 3, 512, 1024])
                input_image_postprocess = sam_model.preprocess(input_image_torch)
                with torch.no_grad():
                    image_embedding = sam_model.image_encoder(input_image_postprocess)

                for curr_gt_mask, curr_bbox, curr_label in zip(gt_mask, bboxes, labels):
                    binary_mask = compute_loss_and_mask(sam_model, image_embedding, curr_bbox, predictor.device, input_size, original_image_size)
                    loss = get_loss(sam_model, binary_mask, curr_gt_mask)
                    iou = calculate_iou(curr_gt_mask, binary_mask)
                    wandb.log({f"Iteration_{iter_num + 1}/Validation/Epoch": epoch, f"Iteration_{iter_num + 1}/Validation/IoU": iou, f"Iteration_{iter_num + 1}/Validation/loss": loss.cpu().numpy().item()})

                    # if iou < 0.5:
                        #   Save the image, gt_mask, and binary_mask
                        # Define the target size
                        # target_size = (1024, 2048)

                    #     # Interpolate the image to the target size
                    #     input_image_torch_vis = input_image_torch.float() / 255.0
                    #     input_image_torch_vis = F.interpolate(input_image_torch_vis, size=target_size, mode='bilinear', align_corners=False)

                        
                    #     visualize_and_save(input_image_torch_vis, curr_gt_mask, binary_mask, curr_bbox, filename=f"valid_set_{curr_label}_iou_{iou}.png")
                    
                    validation_loss.append(loss.item())

                    # Calculate class-wise losses and IoUs:
                    
                    for label in labels:
                        label_str = str(label)
                        if label_str not in class_losses:
                            class_losses[label_str] = []
                            class_ious[label_str] = []
                        class_losses[label_str].append(loss.cpu().numpy().item())
                        class_ious[label_str].append(iou)
        class_stats = {}
        for label_str in class_losses:
            mean_loss = np.mean(class_losses[label_str])
            std_loss = np.std(class_losses[label_str])
            mean_iou = np.mean(class_ious[label_str])
            std_iou = np.std(class_ious[label_str])
            
            class_stats[label_str] = {
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "mean_iou": mean_iou,
                "std_iou": std_iou
            }

        # # Log the results to wandb
        # wandb.log({"epoch": epoch, "class_stats": class_stats})

        
        # # Convert dictionary to DataFrame
        # df = pd.DataFrame(class_stats)
        
        # # Format DataFrame to 3 decimal points
        # df = df.round(3)
        
        # # Create a matplotlib figure
        # plt.figure(figsize=(10, 4))
        
        # # Create a table plot
        # sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, linewidths=.5)
        
        # # Save the plot as a PNG file
        # plt.savefig('class_stats_table.png')
        
        sam_model.train()
        return np.mean(validation_loss)
    
    best_val_loss = float('inf')
    # best_val_loss = 100000
    epochs_without_improvement = 0
    

    for epoch in range(epoches):
        singleImageLoggingFlag = True
        for index, (input_image, data) in enumerate(train_dataset):
            (input_image, data) = train_dataset[0]

            ######### Demo Sanity Check - 1 ##########

            # input_demo = input_image
            # predictor_demo, _ = setup_sam_model()

            ##########################################

            if data is None:
                continue
            
            gt_mask, bboxes, labels = get_values_from_data_iter(data, batch_size, predictor)
            
            if labels == None:
                    continue
            if len(labels) == 0:
                continue

            input_image = input_image.to(predictor.device)
            original_image_size = (1024, 2048)
            input_size = (512, 1024)

            input_image = input_image.permute(1, 2 , 0)
            input_image = input_image.cpu().numpy()
            input_image = input_image * 255
            input_image = input_image.astype(np.uint8)
            input_image = predictor.transform_image(input_image) # change shape 2048 --> 1024
            input_image_torch = torch.as_tensor(input_image, device=predictor.device) # uint8 0:255 values - torch.Size([512, 1024, 3]) 
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :] # uint8 0:255 values - torch.Size([1, 3, 512, 1024])
            input_image_postprocess = sam_model.preprocess(input_image_torch)
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image_postprocess)

   
            for i, (curr_gt_mask, curr_bbox, curr_label) in enumerate(zip(gt_mask, bboxes, labels)):
                
            #     ########## Demo Sanity Check - 2 ##########

            #     # mask_demo = sam_demo_code(input_demo, curr_bbox, predictor_demo)

            #     ###########################################


                binary_mask = compute_loss_and_mask(sam_model, image_embedding, curr_bbox, predictor.device, input_size, original_image_size)
                loss = get_loss(sam_model, binary_mask, curr_gt_mask)
                optimizer.zero_grad()
                loss.backward()
                            
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in sam_model.parameters() if p.grad is not None])) # Calculate gradient norm before clipping
                wandb.log({f"Iteration_{iter_num + 1}/Epoch": epoch, f"Iteration_{iter_num + 1}/Grad Norm Before Clipping": total_norm.item(), f"Iteration_{iter_num + 1}/train_loss": loss.item()})

                clip_grad_norm_(sam_model.parameters(), max_grad_norm) # Apply gradient clippinp
                optimizer.step()
                
            #     # if singleImageLoggingFlag:
            #     #     # Define the target size
            #     #     target_size = (1024, 2048)

            #     #     # Interpolate the image to the target size
            #     #     input_image_torch = input_image_torch.float() / 255.0
            #     #     input_image_torch = F.interpolate(input_image_torch, size=target_size, mode='bilinear', align_corners=False)

            #     #     visualize_and_save(input_image_torch, curr_gt_mask, binary_mask, curr_bbox, filename="test_sam.png")
            #     #     # visualize_and_save(input_image_torch, curr_gt_mask, mask_demo[None,:,:], curr_bbox, filename="test_sam_demo.png")
            #     #     wandb.log({f"Iteration_{iter_num + 1}/Validation Image | epoch {epoch}": [wandb.Image("test_sam.png")]})
            #     #     #singleImageLoggingFlag = False

            del loss, binary_mask, image_embedding, input_image_postprocess
            torch.cuda.empty_cache()


                
        val_loss = validate_model(validation_dataset, epoch, iter_num)

        current_lr = optimizer.param_groups[0]['lr']
 
        wandb.log({f"Iteration_{iter_num + 1}/Epoch": epoch, 
                   f"Iteration_{iter_num + 1}/Validation Loss": val_loss, 
                   f"Iteration_{iter_num + 1}/lr": current_lr})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Apply warmup scheduler if in warmup phase, otherwise apply StepLR
        if epoch < warmup_steps:
            warmup_scheduler.step()
        else:
            scheduler.step()

    # Save model to disk
    torch.save(sam_model.state_dict(), "/workspace/sam_al/model-directory/fine_tune_sam_model.pth")
    return best_val_loss
