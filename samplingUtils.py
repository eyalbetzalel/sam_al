import torch
from torch.utils.data import Dataset, Subset
import random

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
    
    def __init__(self, dataset, train_percent, sampling_method='random', fixed_index=0):
        self.dataset = dataset
        self.train_percent = train_percent
        self.sampling_method = sampling_method
        self.fixed_index = fixed_index
        self.labeled_indices = self.sample_indices()
        self.unlabeled_indices = list(set(range(len(dataset))) - set(self.labeled_indices))

    def sample_indices(self):
        """ Sample indices based on the specified method. """
        total_samples = len(self.dataset)
        num_train_samples = int(total_samples * self.train_percent)

        if self.sampling_method == 'random':
            return random.sample(range(total_samples), num_train_samples)
        elif self.sampling_method == 'fixed':
            return [self.fixed_index]
        else:
            raise ValueError("Unsupported sampling method")

    def get_training_subset(self):
        """ Returns a Subset of the dataset for training based on sampled indices. """
        return Subset(self.dataset, self.labeled_indices)

    def get_unlabeled_subset(self):
        """ Returns a Subset of the dataset for unlabeled samples. """
        return Subset(self.dataset, self.unlabeled_indices)
    
    def update_labeled_set(self, new_indices):
        """ Move samples from unlabeled to labeled set. """
        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = list(set(self.unlabeled_indices) - set(new_indices))
