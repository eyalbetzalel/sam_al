import torch
import random
from torch.utils.data import Subset
from model import DiceLoss, setup_sam_model, finetune_sam_model
from utils import get_values_from_data_iter, visualize_and_save
from metrics import calculate_iou
import wandb
import numpy as np

class ActiveLearningPlatform:
    def __init__(self, model, predictor, initial_train_dataset, val_dataset, test_dataset, batch_size, training_epoch_per_iteration, lr, max_active_learning_iterations, query_strategy):
        """
        Initialize the Active Learning Platform.

        Parameters:
        model (SAM): The SAM model.
        predictor (SamPredictor): The SAM predictor.
        initial_train_dataset (Dataset): The initial training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The test dataset.
        batch_size (int): Batch size for training.
        training_epoch_per_iteration (int): Max number of training epochs per iteration (train will stop if there is no improvement in valdation loss).
        lr (float): Learning rate for training.
        max_active_learning_iterations (int): Maximum number of active learning iterations.
        query_strategy (function): Function to select samples from the unlabeled dataset.
        """
        self.model = model
        self.predictor = predictor
        self.validation_dataset = val_dataset
        # self.validation_dataset = Subset(val_dataset, range(10))
        # self.test_dataset = test_dataset
        self.test_dataset = Subset(test_dataset, range(1000))
        self.batch_size = batch_size
        self.training_epoch_per_iteration = training_epoch_per_iteration
        self.lr = lr
        self.max_iterations = max_active_learning_iterations
        self.query_strategy = query_strategy
        self.active_learning_dataset = ActiveLearningDataset(initial_train_dataset, train_percent=0.1, sampling_method='random')
        # self.active_learning_dataset = ActiveLearningDataset(initial_train_dataset, train_percent=0.1, sampling_method='fixed')s
        
    def train_model(self):
        """
        Train the model on the current labeled dataset.
        """

        training_subset = self.active_learning_dataset.get_training_subset()
        finetune_sam_model(self.model, self.predictor, training_subset, self.validation_dataset, batch_size=self.batch_size, epoches=self.training_epoch_per_iteration, iter_num=self.iteration, lr=self.lr)

    def perform_inference(self):
        """
        Perform inference on the unlabeled dataset (for advanced strategies).
        """
        pass

    def query_labels(self, num_images_to_query=1):
        """
        Query labels from the unlabeled dataset using the query strategy.

        Returns:
        list: Indices of selected samples from the unlabeled dataset.
        """
        unlabeled_subset = self.active_learning_dataset.get_unlabeled_subset()
        num_samples_to_query = min(len(unlabeled_subset), num_images_to_query)
        queried_indices = self.query_strategy(unlabeled_subset, num_samples_to_query)
        return queried_indices

    def update_datasets(self, new_indices):
        """
        Update the datasets by moving newly labeled samples from the unlabeled to labeled set.

        Parameters:
        new_indices (list): Indices of newly labeled samples.
        """
        self.active_learning_dataset.update_labeled_set(new_indices)

    def compute_loss_and_mask(self, image_embedding, bbox):
        """
        Compute the loss and mask for a given image embedding and bounding box.

        Parameters:
        image_embedding (torch.Tensor): Image embedding tensor.
        bbox (torch.Tensor): Bounding box tensor.

        Returns:
        torch.Tensor: Binary mask.
        """
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=bbox, masks=None)

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding, 
            image_pe=self.model.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings, 
            dense_prompt_embeddings=dense_embeddings, 
            multimask_output=False
        )

        upscaled_masks = self.model.postprocess_masks(low_res_masks, (512, 1024), (1024, 2048)).to(self.predictor.device)
        binary_mask = torch.nn.functional.normalize(torch.nn.functional.threshold(upscaled_masks, 0.0, 0)).to(self.predictor.device)
        binary_mask = binary_mask.squeeze()
        if len(binary_mask.shape) == 2:
            binary_mask = binary_mask.unsqueeze(0)

        return binary_mask

    def get_loss(self, binary_mask, gt_mask):
        """
        Compute the loss between the binary mask and ground truth mask.

        Parameters:
        binary_mask (torch.Tensor): Binary mask tensor.
        gt_mask (torch.Tensor): Ground truth mask tensor.

        Returns:
        torch.Tensor: Loss value.
        """
        # loss_fn = DiceLoss(smooth=1)
        loss_fn = torch.nn.MSELoss()
        return loss_fn(binary_mask, gt_mask.float())
    
    def test_model(self):
        """
        Test the model on the test dataset.

        Returns:
        tuple: Average test loss and average IoU score.
        """
        self.model.eval()
        test_loss = []
        iou_scores = []
        singleImageLoggingFlag = True

        with torch.no_grad():

            for input_image, data in self.test_dataset:
                if data is None:
                    continue
                gt_mask, bboxes, labels = get_values_from_data_iter(data, self.batch_size, self.predictor)
                if len(labels) == 0:
                    continue
                input_image = input_image.to(self.predictor.device)
                input_image = input_image.unsqueeze(0)
                
                for curr_gt_mask, curr_bbox in zip(gt_mask, bboxes):
                    input_image_postprocess = self.model.preprocess(input_image)
                    image_embedding = self.model.image_encoder(input_image_postprocess)
                    binary_mask = self.compute_loss_and_mask(image_embedding, curr_bbox)
                    loss = self.get_loss(binary_mask, curr_gt_mask)
                    test_loss.append(loss.item())
                    
                    iou_score = calculate_iou(curr_gt_mask.cpu().numpy(), binary_mask.cpu().numpy())
                    iou_scores.append(iou_score)

                    if singleImageLoggingFlag:
                        visualize_and_save(input_image, curr_gt_mask, binary_mask, curr_bbox, filename="test_sam.png")
                        wandb.log({f"Iteration_{self.iteration + 1}/Test Image": [wandb.Image("test_sam.png")]})
                        singleImageLoggingFlag = False

            avg_test_loss = np.mean(test_loss)
            avg_iou = np.mean(iou_scores)
            print(f"Test Loss: {avg_test_loss}, Average IoU: {avg_iou}")
            return avg_test_loss, avg_iou

    def run(self, precent_from_dataset_to_query_each_iteration=0.1):
        """
        Run the active learning process.

        Parameters:
        precent_from_dataset_to_query_each_iteration (float): Percentage of the dataset to query in each iteration.

        Returns:
        None
        """
        wandb.init(
            project="ActiveLearningSAM",
        )
        num_images_to_query = int(np.floor(precent_from_dataset_to_query_each_iteration * len(self.active_learning_dataset.dataset)))
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            # if iteration != 0:
            self.train_model()
            print(f"Iteration {iteration}/{self.max_iterations} complete")    
            
            
            total_precent_so_far = precent_from_dataset_to_query_each_iteration * iteration
            # test_loss, test_iou = self.test_model()
            # wandb.log({"Test IoU": test_iou, 
            #            "Test Loss": test_loss,
            #            "[%] Dataset": total_precent_so_far})
            wandb.log({"[%] Dataset": total_precent_so_far})
            num_images_to_query = int(np.floor(precent_from_dataset_to_query_each_iteration * len(self.active_learning_dataset.dataset)))
            queried_indices = self.query_labels(num_images_to_query)
            self.update_datasets(queried_indices)
          
        torch.save(self.model.state_dict(), "model-directory/final_sam_model.pth")
        wandb.finish()

def random_query_strategy(unlabeled_subset, num_samples):
    """
    Randomly select samples from the unlabeled dataset.

    Parameters:
    unlabeled_subset (Subset): Unlabeled subset of the dataset.
    num_samples (int): Number of samples to select.

    Returns:
    list: Indices of selected samples.
    """
    return random.sample(range(len(unlabeled_subset)), num_samples)

class ActiveLearningDataset:
    """
    A class to manage labeled and unlabeled datasets for active learning.

    Attributes:
        dataset (Dataset): The complete dataset.
        train_percent (float): The percentage of the dataset to initially label.
        sampling_method (str): The method used to sample the initial labeled data.
        fixed_index (int): A fixed index for certain sampling methods.
        labeled_indices (list): Indices of the labeled dataset.
        unlabeled_indices (list): Indices of the unlabeled dataset.
    """
    
    def __init__(self, dataset, train_percent, sampling_method='random', fixed_index=7):
        self.dataset = dataset
        self.train_percent = train_percent
        self.sampling_method = sampling_method
        self.fixed_index = fixed_index
        self.labeled_indices = self.sample_indices()
        self.unlabeled_indices = list(set(range(len(dataset))) - set(self.labeled_indices))

    def sample_indices(self):
        """
        Sample indices based on the specified method.

        Returns:
        list: List of sampled indices.
        """
        total_samples = len(self.dataset)
        num_train_samples = int(total_samples * self.train_percent)

        if self.sampling_method == 'random':
            return random.sample(range(total_samples), num_train_samples)
        elif self.sampling_method == 'fixed':
            return [self.fixed_index]
        else:
            raise ValueError("Unsupported sampling method")

    def get_training_subset(self):
        """
        Returns a Subset of the dataset for training based on sampled indices.

        Returns:
        Subset: Subset of the dataset for training.
        """
        return Subset(self.dataset, self.labeled_indices)

    def get_unlabeled_subset(self): 
        """
        Returns a Subset of the dataset for unlabeled samples.

        Returns:
        Subset: Subset of the dataset for unlabeled samples.
        """
        return Subset(self.dataset, self.unlabeled_indices)
    
    def update_labeled_set(self, new_indices):
        """
        Move samples from unlabeled to labeled set.

        Parameters:
        new_indices (list): List of new indices to add to the labeled set.
        """
        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = list(set(self.unlabeled_indices) - set(new_indices))
