import os 
import yaml
import pickle

import matplotlib.pyplot as plt

class Config:
    def __init__(self, config_dict):
        self.experiment = config_dict.get("experiment", {})
        self.data = config_dict.get("data", {})
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})

    def __reper__(self):
        return f"Config(experiment={self.experiment}, data={self.data} model={self.model}, training={self.training})"
     

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = Config(config)
    return config


def save_checkpoint(model, optimizer, epoch, val_acc, config, exp_dir, epoch_num):
    # Saving model as a pickle file at the end of each epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config,
    }
    checkpoint_path = os.path.join(exp_dir, f"checkpoint_epoch_{epoch_num}.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at {checkpoint_path}")

def lr_vs_epoch(num_epochs, lrs, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), lrs, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs. Epoch")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_rate_plot.png'), bbox_inches='tight')
