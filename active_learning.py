import torch
import random
from torch.utils.data import Subset
from model import DiceLoss, setup_sam_model, finetune_sam_model
from utils import get_values_from_data_iter, visualize_and_save
from metrics import calculate_iou  # Assuming calculate_iou is defined here
import wandb
import numpy as np
import torch.nn.functional as F

class ActiveLearningPlatform:
    def __init__(self, model, predictor, initial_train_dataset, val_dataset, test_dataset, batch_size, training_epoch_per_iteration, lr, max_active_learning_iterations, query_strategy, optimizer_name='Adam', warmup_steps=5, gamma=0.8, step_size=5):
        # Existing initialization code...
        self.model = model
        self.predictor = predictor
        self.validation_dataset = Subset(val_dataset, range(50)) #val_dataset 
        initial_train_dataset = Subset(initial_train_dataset, range(100)) #initial_train_dataset 
        self.test_dataset = Subset(test_dataset, range(10))
        self.batch_size = batch_size
        self.training_epoch_per_iteration = training_epoch_per_iteration
        self.lr = lr
        self.max_iterations = max_active_learning_iterations
        self.query_strategy = query_strategy
        self.active_learning_dataset = ActiveLearningDataset(initial_train_dataset, train_percent=0.001, sampling_method='random')
        self.loss_fn = torch.nn.MSELoss()  # Initialize the loss function
        self.optimizer_name = optimizer_name
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.step_size = step_size
        self.best_val_loss = None
        self.training_step = 0
        self.validation_step = 0

    def train_model(self):
        training_subset = self.active_learning_dataset.get_training_subset()
        self.best_val_loss, self.training_step, self.validation_step = finetune_sam_model(
            sam_model=self.model,
            predictor=self.predictor,
            train_dataset=training_subset,
            validation_dataset=self.validation_dataset,
            batch_size=self.batch_size,
            epoches=self.training_epoch_per_iteration,
            patience=3,
            iter_num=self.iteration,
            lr=self.lr,
            optimizer_name=self.optimizer_name,
            warmup_steps=self.warmup_steps,
            gamma=self.gamma,
            step_size=self.step_size,
            training_step=self.training_step,
            validation_step=self.validation_step
        )

    def get_validation_loss(self):
        return self.best_val_loss

    def compute_loss_and_mask(self, image_embedding, bbox, input_size, original_image_size, delta=0.1):
        # Updated compute_loss_and_mask method with mask clipping
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=bbox, masks=None)

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        upscaled_masks = self.model.postprocess_masks(low_res_masks, input_size, original_image_size).to(self.predictor.device)
        binary_mask = torch.nn.functional.normalize(torch.nn.functional.threshold(upscaled_masks, 0.0, 0)).to(self.predictor.device)
        binary_mask = binary_mask.squeeze()
        if len(binary_mask.shape) == 2:
            binary_mask = binary_mask.unsqueeze(0)

        # Clip the mask to the bounding box with an extension delta
        binary_mask = self.clip_mask_to_bbox(binary_mask, bbox, delta)

        return binary_mask

    def clip_mask_to_bbox(self, mask, bbox, delta=0.1):
        # Clipping function as described
        height, width = mask.shape[-2:]
        bbox = bbox.cpu().numpy().astype(int).tolist()[0]

        x_min, y_min, x_max, y_max = 2 * bbox[0], 2 * bbox[1], 2 * bbox[2], 2 * bbox[3]
        box_width = x_max - x_min
        box_height = y_max - y_min

        x_min_extended = max(0, x_min - delta * box_width)
        y_min_extended = max(0, y_min - delta * box_height)
        x_max_extended = min(width, x_max + delta * box_width)
        y_max_extended = min(height, y_max + delta * box_height)

        clipped_mask = torch.zeros_like(mask)
        clipped_mask[:, int(y_min_extended):int(y_max_extended), int(x_min_extended):int(x_max_extended)] = mask[:, int(y_min_extended):int(y_max_extended), int(x_min_extended):int(x_max_extended)]

        return clipped_mask

    def get_loss(self, binary_mask, gt_mask):
        # Updated get_loss method using the initialized loss function
        return self.loss_fn(binary_mask, gt_mask.float())

    def compute_iou_scores(self, dataset):
        """
        Perform inference on the dataset and compute average IoU scores for each image.

        Parameters:
        dataset (Dataset): The dataset to perform inference on.

        Returns:
        dict: Dictionary mapping indices in the dataset to average IoU scores.
        """
        self.model.eval()
        iou_scores = {}
        input_size = (512, 1024)
        original_image_size = (1024, 2048)
        with torch.no_grad():
            for idx_in_subset, (input_image, data) in enumerate(dataset):
                if data is None:
                    continue
                gt_masks, bboxes, labels = get_values_from_data_iter(data, self.batch_size, self.predictor)
                
                if labels == None:
                    continue
                if len(labels) == 0:
                    continue

                input_image = input_image.to(self.predictor.device)
                input_image = input_image.permute(1, 2, 0)
                input_image = input_image.cpu().numpy()
                input_image = input_image * 255
                input_image = input_image.astype(np.uint8)
                input_image = self.predictor.transform_image(input_image)
                input_image_torch = torch.as_tensor(input_image, device=self.predictor.device)
                input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
                input_image_postprocess = self.model.preprocess(input_image_torch)
                with torch.no_grad():
                    image_embedding = self.model.image_encoder(input_image_postprocess)

                image_iou_scores = []
                for curr_gt_mask, curr_bbox in zip(gt_masks, bboxes):
                    binary_mask = self.compute_loss_and_mask(image_embedding, curr_bbox, input_size, original_image_size)
                    iou_score = calculate_iou(curr_gt_mask.cpu().numpy(), binary_mask.cpu().numpy())
                    image_iou_scores.append(iou_score)

                avg_iou = np.mean(image_iou_scores)
                # Map the index to the average IoU
                iou_scores[idx_in_subset] = avg_iou  # idx_in_subset is index in unlabeled_subset
        return iou_scores

    def oracle_query_strategy(self, unlabeled_subset, num_samples):
        """
        Query strategy that uses ground truth as oracle to select samples with lowest IoU scores.

        Parameters:
        unlabeled_subset (Subset): Unlabeled subset of the dataset.
        num_samples (int): Number of samples to select.

        Returns:
        list: Indices of selected samples from the unlabeled dataset (indices into the dataset).
        """
        # Compute IoU scores for each sample in the unlabeled_subset
        iou_scores = self.compute_iou_scores(unlabeled_subset)

        # Sort the indices by IoU score (from low to high)
        sorted_indices_in_subset = sorted(iou_scores, key=iou_scores.get)

        # Select the indices of the num_samples lowest IoU scores
        
        # ################## TODO : DELETE THIS LINE ##################
        # if num_samples == 0:
        #     num_samples = 2
        # ################## TODO : DELETE THIS LINE ##################

        selected_indices_in_subset = sorted_indices_in_subset[:num_samples]

        # Map indices in the unlabeled_subset back to indices in the dataset
        selected_indices_in_dataset = [unlabeled_subset.indices[idx] for idx in selected_indices_in_subset]

        return selected_indices_in_dataset

    def random_query_strategy(self, unlabeled_subset, num_samples):
        return random.sample(range(len(unlabeled_subset)), num_samples)

    def query_labels(self, num_images_to_query=1):
        """
        Query labels from the unlabeled dataset using the query strategy.

        Returns:
        list: Indices of selected samples from the unlabeled dataset.
        """
        unlabeled_subset = self.active_learning_dataset.get_unlabeled_subset()
        num_samples_to_query = min(len(unlabeled_subset), num_images_to_query)

        if self.query_strategy == 'random':
            queried_indices = self.random_query_strategy(unlabeled_subset, num_samples_to_query)
            queried_indices = [unlabeled_subset.indices[idx] for idx in queried_indices]  # Map back to dataset indices
        elif self.query_strategy == 'oracle':
            queried_indices = self.oracle_query_strategy(unlabeled_subset, num_samples_to_query)
        else:
            # If self.query_strategy is a function, call it
            queried_indices = self.query_strategy(unlabeled_subset, num_samples_to_query)
        return queried_indices

    def test_model(self):
        # Updated test_model method as previously described
        self.model.eval()
        test_loss = []
        iou_scores = []
        singleImageLoggingFlag = True

        input_size = (512, 1024)
        original_image_size = (1024, 2048)
        with torch.no_grad():
            for input_image, data in self.test_dataset:
                if data is None:
                    continue
                gt_mask, bboxes, labels = get_values_from_data_iter(data, self.batch_size, self.predictor)
                if len(labels) == 0:
                    continue

                input_image = input_image.to(self.predictor.device)
                input_image = input_image.permute(1, 2, 0)
                input_image = input_image.cpu().numpy()
                input_image = input_image * 255
                input_image = input_image.astype(np.uint8)
                input_image = self.predictor.transform_image(input_image)
                input_image_torch = torch.as_tensor(input_image, device=self.predictor.device)
                input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
                input_image_postprocess = self.model.preprocess(input_image_torch)
                with torch.no_grad():
                    image_embedding = self.model.image_encoder(input_image_postprocess)

                for curr_gt_mask, curr_bbox in zip(gt_mask, bboxes):
                    binary_mask = self.compute_loss_and_mask(image_embedding, curr_bbox, input_size, original_image_size)
                    loss = self.get_loss(binary_mask, curr_gt_mask)
                    test_loss.append(loss.item())

                    iou_score = calculate_iou(curr_gt_mask.cpu().numpy(), binary_mask.cpu().numpy())
                    iou_scores.append(iou_score)

                    if singleImageLoggingFlag:
                        # Define the target size
                        target_size = (1024, 2048)
                        # Interpolate the image to the target size
                        input_image_torch = input_image_torch.float() / 255.0
                        input_image_torch = F.interpolate(input_image_torch, size=target_size, mode='bilinear', align_corners=False)

                        visualize_and_save(input_image_torch, curr_gt_mask, binary_mask, curr_bbox, filename="test_sam.png")
                        wandb.log({f"Iteration_{self.iteration + 1}/Test Image": [wandb.Image("test_sam.png")]})
                        singleImageLoggingFlag = False

            avg_test_loss = np.mean(test_loss)
            avg_iou = np.mean(iou_scores)
            print(f"Test Loss: {avg_test_loss}, Average IoU: {avg_iou}")
            return avg_test_loss, avg_iou

    def run(self, precent_from_dataset_to_query_each_iteration=0.1):
        # Updated run method as previously described
        # Create a unique run name with hyperparameters
        run_name = (
            f"Trial_lr_{self.lr}_bs_{self.batch_size}_opt_{self.optimizer_name}"
            f"_ws_{self.warmup_steps}_gamma_{self.gamma}_step_{self.step_size}"
        )
        wandb.init(
            project="ActiveLearningSAM",
            name=run_name,
            config={
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "optimizer": self.optimizer_name,
                "warmup_steps": self.warmup_steps,
                "gamma": self.gamma,
                "step_size": self.step_size
            }
            # mode="disabled" 
        )
        # Initialize step counters at the beginning of each run
        self.training_step = 0
        self.validation_step = 0
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            # if iteration > 0:
            #     self.train_model()
                

            total_percent_so_far = precent_from_dataset_to_query_each_iteration * iteration
            # test_loss, test_iou = self.test_model()
            # wandb.log({
            #     "Test IoU": test_iou, 
            #     "Test Loss": test_loss,
            #     "[%] Dataset": total_percent_so_far
            # })

            num_images_to_query = int(np.floor(precent_from_dataset_to_query_each_iteration * len(self.active_learning_dataset.dataset)))
            queried_indices = self.query_labels(num_images_to_query)
            self.active_learning_dataset.update_labeled_set(queried_indices)
            self.train_model()
            print(f"Iteration {iteration}/{self.max_iterations} complete")
          
        torch.save(self.model.state_dict(), "model-directory/final_sam_model.pth")
        wandb.finish()

# The random query strategy remains the same


# ActiveLearningDataset class remains the same
class ActiveLearningDataset:
    # Existing code...
    def __init__(self, dataset, train_percent, sampling_method='random', fixed_index=7):
        self.dataset = dataset
        self.train_percent = train_percent
        self.sampling_method = sampling_method
        self.fixed_index = fixed_index
        self.labeled_indices = self.sample_indices()
        self.unlabeled_indices = list(set(range(len(dataset))) - set(self.labeled_indices))

    def sample_indices(self):
        total_samples = len(self.dataset)
        num_train_samples = int(total_samples * self.train_percent)

        if self.sampling_method == 'random':
            return random.sample(range(total_samples), num_train_samples)
        elif self.sampling_method == 'fixed':
            return [self.fixed_index]
        else:
            raise ValueError("Unsupported sampling method")

    def get_training_subset(self):
        return Subset(self.dataset, self.labeled_indices)

    def get_unlabeled_subset(self): 
        return Subset(self.dataset, self.unlabeled_indices)
    
    def update_labeled_set(self, new_indices):
        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = list(set(self.unlabeled_indices) - set(new_indices))
