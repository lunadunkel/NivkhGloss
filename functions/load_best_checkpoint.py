import os
import glob
import torch

def load_best_checkpoint(model, optimizer, checkpoint_dir="checkpoints", device="cpu"):
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' does not exist.")

    list_of_files = glob(os.path.join(checkpoint_dir, "best_model_*.pt"))

    if not list_of_files:
        raise FileNotFoundError("No checkpoint files found in directory")

    best_checkpoint_path = max(list_of_files, key=os.path.getctime)

    checkpoint = torch.load(best_checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from: {best_checkpoint_path}")
    return model, optimizer, best_checkpoint_path