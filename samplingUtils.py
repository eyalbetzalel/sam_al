import torch
from torch.utils.data import Dataset, Subset
import random

class ActiveLearningDataset:
    
    def __init__(self, dataset, train_percent, sampling_method='random'):
        self.dataset = dataset
        self.train_percent = train_percent
        self.sampling_method = sampling_method
        self.train_indices = self.sample_indices()

    def sample_indices(self):
        """ Sample indices based on the specified method. """
        total_samples = len(self.dataset)
        num_train_samples = int(total_samples * self.train_percent)

        if self.sampling_method == 'random':
            return random.sample(range(total_samples), num_train_samples)
        else:
            raise ValueError("Unsupported sampling method")

    def get_training_subset(self):
        """ Returns a Subset of the dataset for training based on sampled indices. """
        return Subset(self.dataset, self.train_indices)
