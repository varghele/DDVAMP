import torch
import os
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
            self,
            save_path: str,
            file_name: str = 'best_network',
            patience: int = 15,
            verbose: bool = False,
            delta: float = 1e-6
    ):
        """
        Initialize early stopping parameters.

        Args:
            save_path: Directory path to save the model
            file_name: Name of the saved model file
            patience: Number of epochs to wait before early stopping
            verbose: If True, prints validation loss improvements
            delta: Minimum change in monitored value to qualify as improvement
        """
        self.save_path = save_path
        self.file_name = file_name
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.val_loss_min = float('inf')
        self.early_stop = False
        self.is_best = False

    def __call__(self, val_loss: float, model) -> None:
        """
        Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss
            model: Model to save if validation loss improves
        """
        score = val_loss

        if np.isnan(score):
            self.is_best = False
            print(f'NaN detected! EarlyStopping counter: {self.counter}/{self.patience}')
            self.early_stop = True
            return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.is_best = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.is_best = False
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.is_best = True

    def save_checkpoint(self, val_loss: float, model) -> None:
        """
        Save model when validation loss decreases.

        Args:
            val_loss: Current validation loss
            model: Model to save
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        path = os.path.join(self.save_path, f"{self.file_name}.pt")

        if isinstance(model, dict):
            torch.save(model, path)
        else:
            torch.save(model.state_dict(), path)

        self.val_loss_min = val_loss
