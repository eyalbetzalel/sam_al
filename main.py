from model import setup_sam_model
from active_learning import ActiveLearningPlatform, random_query_strategy
from importCityScapesToDataloader import train_dataset, val_dataset, test_dataset

# Example Usage
batch_size = 1
max_iterations = 7
query_strategy = random_query_strategy
training_epoch_per_iteration = 2000
lr = 1e-5

# Setup SAM model and predictor
predictor, sam_model = setup_sam_model()

# Initialize the Active Learning Platform
active_learning_platform = ActiveLearningPlatform(
    sam_model, 
    predictor, 
    train_dataset,
    val_dataset, 
    test_dataset, 
    batch_size,
    training_epoch_per_iteration,
    lr, 
    max_iterations, 
    query_strategy
)

# Run the active learning process
active_learning_platform.run(precent_from_dataset_to_query_each_iteration=0.1)

# TODO: Implement a more advanced query strategy for active learning.
# TODO: Implement a method to load a trained model from disk.
# TODO: After validating that logging is work - change input size before night run
