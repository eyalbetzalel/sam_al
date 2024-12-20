import optuna
import time  
from model import setup_sam_model
import wandb  
from active_learning import ActiveLearningPlatform
from importCityScapesToDataloader import train_dataset, val_dataset, test_dataset

batch_size = 16
max_iterations = 9
query_strategy = 'oracle'
training_epoch_per_iteration = 100
lr = 1e-5

def objective(trial):
    lr = trial.suggest_float('lr', 1e-7, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    warmup_steps = trial.suggest_int('warmup_steps', 1, 3)
    gamma = trial.suggest_float('gamma', 0.1, 0.9)
    step_size = trial.suggest_int('step_size', 1, 3)

    predictor, sam_model = setup_sam_model()

    active_learning_platform = ActiveLearningPlatform(
        sam_model, 
        predictor, 
        train_dataset,
        val_dataset, 
        test_dataset, 
        batch_size,
        training_epoch_per_iteration=10,
        lr=lr, 
        max_active_learning_iterations=1, 
        query_strategy='oracle',
        optimizer_name=optimizer_name,
        warmup_steps=warmup_steps,
        gamma=gamma,
        step_size=step_size
    )

    start_time = time.time()  # Start time
    active_learning_platform.run(precent_from_dataset_to_query_each_iteration=0.1)
    end_time = time.time()  # End time

    elapsed_time = end_time - start_time

    val_loss = active_learning_platform.get_validation_loss()

    print(f"Trial completed in {elapsed_time:.2f} seconds")

    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Log output to a file
    with open("output.log", "w") as f:
        f.write(f"Study best trial: {study.best_trial}\n")
        f.write(f"Best value: {study.best_value}\n")
        f.write(f"Best parameters: {study.best_params}\n")

# TODO: Implement a more advanced query strategy for active learning.
# TODO: Implement a method to load a trained model from disk.
# TODO: After validating that logging is work - change input size before night run
