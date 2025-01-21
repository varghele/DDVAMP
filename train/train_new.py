import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from args import buildParser
from utils.unflatten import unflatten
from utils.count_parameters import count_parameters

from deeptime.util.data import TrajectoryDataset
import numpy as np
from components.models.GraphVAMPNet import GraphVampNet
from components.models.RevVAMPNet import RevVAMPNet
from train.utils.EarlyStopping import EarlyStopping
from tqdm import tqdm
from typing import Optional, Tuple
from components.models.vamps import VAMPS
from components.models.vampu import VAMPU


def flush_cuda_cache():
    """Flush CUDA cache and clear memory."""
    try:
        # Empty CUDA cache
        torch.cuda.empty_cache()

        # Synchronize CUDA streams
        torch.cuda.synchronize()

        # Optional: Force garbage collection
        #import gc
        #gc.collect()

    except Exception as e:
        print(f"Warning: Could not flush CUDA cache: {str(e)}")

def record_result(arg1, arg2):
    pass

class ExpActivation(nn.Module):
    """Exponential activation function as a proper nn.Module."""
    def forward(self, x):
        return torch.exp(x)

class RevVAMPTrainer:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.setup_directories()

        # Initialize data
        self.file_path = '../datasets/traj_revgraphvamp_org/intermediate/'
        self.traj_length, self.dists1, self.inds1 = self._load_data()
        self.dataset = self._prepare_dataset()
        self.loader_train, self.loader_val, self.loader_train_all = self._create_dataloaders()

        # Initialize model
        self.lobe, self.vampnet, self.vlu, self.vls = self._initialize_model()

    def _setup_device(self):
        """Set up and return the appropriate device (CPU/CUDA)."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('CUDA is available')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def setup_directories(self):
        """Create save directories and log arguments."""
        self.log_pth = f'{self.args.save_folder}/training.log'
        if not os.path.exists(self.args.save_folder):
            print('Creating folder for saving checkpoints...')
            os.makedirs(self.args.save_folder)

        with open(f'{self.args.save_folder}/args.txt', 'w') as f:
            f.write(str(self.args))

    def _load_data(self):
        """Load and prepare data from files."""
        data_info = np.load(f"{self.file_path}red_5nbrs_1ns_datainfo_min.npy", allow_pickle=True).item()
        traj_length = data_info['length']
        dists1 = np.load(f"{self.file_path}red_5nbrs_1ns_dist_min.npy")
        inds1 = np.load(f"{self.file_path}red_5nbrs_1ns_inds_min.npy")
        return traj_length, dists1, inds1

    def _prepare_dataset(self):
        """Prepare dataset from distance and index data."""
        dist_sp = [r for r in unflatten(self.dists1, self.traj_length)]
        inds_sp = [r for r in unflatten(self.inds1, self.traj_length)]

        data = []
        for i in range(len(dist_sp)):
            mydists1 = torch.from_numpy(dist_sp[i])
            myinds1 = torch.from_numpy(inds_sp[i])
            data.append(torch.cat((mydists1, myinds1), axis=-1))

        return TrajectoryDataset.from_trajectories(lagtime=self.args.tau, data=data)

    def _create_dataloaders(self):
        """Create training and validation dataloaders."""
        n_val = int(len(self.dataset) * self.args.val_frac)
        train_data, val_data = torch.utils.data.random_split(
            self.dataset,
            [len(self.dataset) - n_val, n_val]
        )

        loader_train = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)
        loader_val = DataLoader(val_data, batch_size=self.args.batch_size, shuffle=False)
        loader_train_all = DataLoader(train_data, batch_size=len(self.dataset), shuffle=True)

        return loader_train, loader_val, loader_train_all

    def _initialize_model(self):
        """Initialize the RevVAMPNet model and its components."""
        lobe = GraphVampNet()
        lobe_timelagged = deepcopy(lobe).to(device=self.device)
        lobe = lobe.to(self.device)

        print('Number of parameters:', count_parameters(lobe))

        vlu = VAMPU(self.args.num_classes, activation=ExpActivation())
        vls = VAMPS(self.args.num_classes, activation=ExpActivation(), renorm=True)

        vampnet = RevVAMPNet(
            lobe=lobe,
            lobe_timelagged=lobe_timelagged,
            learning_rate=self.args.lr,
            device=self.device,
            optimizer='Adam',
            score_method=self.args.score_method,
            vampu=vlu,
            vamps=vls
        )

        return lobe, vampnet, vlu, vls

    def train_US(self, n_epochs: int, model_name="best_us"):
        """Train the auxiliary networks (U and S) of the VAMPNet model."""
        n_OOM = 0

        print("Training auxiliary networks...")
        early_stopping = EarlyStopping(
            self.args.save_folder,
            file_name=model_name,
            delta=1e-4,
            patience=100
        )

        # Freeze main network weights
        self.vampnet.lobe.requires_grad_(False)
        self.vampnet.lobe_timelagged.requires_grad_(False)

        # Calculate state probabilities
        data_size = sum(batch[0].shape[0] for batch, _ in self.loader_train)
        state_probs = np.zeros((data_size, int(self.args.num_classes)))
        state_probs_tau = np.zeros((data_size, int(self.args.num_classes)))

        n_iter = 0
        with torch.no_grad():
            for batch_0, batch_t in self.loader_train:
                # Trying to flush CUDA to make training more stable
                torch.cuda.empty_cache()
                flush_cuda_cache()

                batch_size = len(batch_0)

                # Check inputs
                if torch.isnan(batch_0).any() or torch.isnan(batch_t).any():
                    print("Warning: NaN values in batch, skipping")
                    continue

                state_probs[n_iter:n_iter + batch_size] = self.vampnet.transform(batch_0)
                state_probs_tau[n_iter:n_iter + batch_size] = self.vampnet.transform(
                    batch_t, instantaneous=False
                )
                n_iter += batch_size

        self.vampnet.update_auxiliary_weights([state_probs, state_probs_tau], optimize_S=True)

        # Train auxiliary networks
        best_dict = None
        for epoch in tqdm(n_epochs):
            try:
                torch.cuda.empty_cache()
                self.vampnet.train_US([state_probs, state_probs_tau])

                if self.loader_val is not None:
                    with torch.no_grad():
                        scores = [self.vampnet.validate((batch[0].to(self.device), batch[1].to(self.device)))
                                  for batch in self.loader_val]
                        mean_score = torch.mean(torch.stack(scores))

                    model_state = {
                        'epoch': n_epochs,
                        'state_dict': self.lobe.state_dict(),
                        'vlu_dict': self.vlu.state_dict(),
                        'vls_dict': self.vls.state_dict(),
                    }
                    early_stopping(mean_score.item(), model_state)

                    if epoch % 50 == 9:
                        record_result(
                            f"Auxiliary training epoch: {epoch}, "
                            f"validation score: {mean_score.item():.4f}",
                            self.log_pth
                        )

                    if early_stopping.is_best:
                        best_dict = {
                            'vlu_dict': self.vlu.state_dict(),
                            'vls_dict': self.vls.state_dict()
                        }

                    if early_stopping.early_stop:
                        print("Early stopping triggered for auxiliary networks")
                        break

            except RuntimeError as e:
                print(f"Runtime error at epoch {epoch}:")
                print(e, e.args[0])
                n_OOM += 1

        del state_probs
        del state_probs_tau

        print(f"Final VAMPCE score: {early_stopping.val_loss_min:.4f}")

        self.vampnet.lobe.requires_grad_(True)
        self.vampnet.lobe_timelagged.requires_grad_(True)

        if best_dict:
            self.vampnet.vampu.load_state_dict(best_dict['vlu_dict'])
            self.vampnet.vamps.load_state_dict(best_dict['vls_dict'])

        record_result(
            f"Auxiliary training complete - epochs: {n_epochs}, "
            f"final score: {early_stopping.val_loss_min:.4f}",
            self.log_pth
        )

        self.vampnet.set_optimizer_lr(0.2)

    def train_vamp(self) -> Tuple[RevVAMPNet, int]:
        """Train the VAMPNet model with optional VAMPCE pre-training."""
        n_OOM = 0
        pre_epoch = self.args.pre_train_epoch

        if self.args.score_method == 'VAMPCE':
            print("Training vanilla VAMPNet model...")
            self.vampnet.vampu.requires_grad_(False)
            self.vampnet.vamps.requires_grad_(False)
            self.vampnet.score_method = 'VAMP2'

            early_stopping = EarlyStopping(
                save_path=self.args.save_folder,
                file_name='best_pre_lobe',
                delta=1e-4,
                patience=100
            )
            best_dict = None

            # Pre-training loop
            print("Entering Pre-Training")
            for epoch in tqdm(range(pre_epoch)):
                try:
                    for batch_0, batch_t in self.loader_train:
                        torch.cuda.empty_cache()
                        self.vampnet.partial_fit((batch_0.to(self.device), batch_t.to(self.device)))

                    if self.loader_val is not None:
                        with torch.no_grad():
                            scores = [self.vampnet.validate((batch[0].to(self.device), batch[1].to(self.device)))
                                      for batch in self.loader_val]
                            mean_score = torch.mean(torch.stack(scores))

                        early_stopping(mean_score.item(), {'state_dict': self.vampnet.lobe.state_dict()})

                        if early_stopping.is_best:
                            best_dict = self.vampnet.lobe.state_dict()

                        if early_stopping.early_stop:
                            print("Early stopping pre-training")
                            break

                        if epoch % 10 == 9:
                            record_result(
                                f"Pre-training step: {epoch}, validation score: {mean_score.item():.4f}, "
                                f"best score: {early_stopping.val_loss_min:.4f}",
                                self.log_pth
                            )

                except RuntimeError as e:
                    print(f"Epoch {epoch}: Runtime error!")
                    print(e, e.args[0])
                    n_OOM += 1

            self.vampnet.vampu.requires_grad_(True)
            self.vampnet.vamps.requires_grad_(True)
            self.vampnet.score_method = 'VAMPCE'

            if best_dict:
                self.vampnet.lobe.load_state_dict(best_dict)
                self.vampnet.lobe_timelagged.load_state_dict(deepcopy(best_dict))
            print(f"VAMP2 best score: {early_stopping.val_loss_min:.4f}")

        # Train auxiliary networks
        self.train_US(pre_epoch)

        # Train full network
        print("Training full network...")
        all_train_epo = 0
        best_model = None
        early_stopping = EarlyStopping(
            self.args.save_folder,
            file_name='best_allnet',
            delta=1e-4,
            patience=200
        )

        for epoch in tqdm(range(self.args.epochs)):
            if epoch == 100:
                for param_group in self.vampnet.optimizer.param_groups:
                    param_group['lr'] = 0.2

            try:
                batch_count = 0
                for batch_0, batch_t in self.loader_train:
                    torch.cuda.empty_cache()
                    self.vampnet.partial_fit((batch_0.to(self.device), batch_t.to(self.device)))
                    all_train_epo += 1
                    batch_count += 1

                if self.loader_val is not None:
                    with torch.no_grad():
                        scores = [self.vampnet.validate((batch[0].to(self.device), batch[1].to(self.device)))
                                  for batch in self.loader_val]
                        mean_score = torch.mean(torch.stack(scores))

                    self.vampnet._validation_scores.append((self.vampnet._step, mean_score.item()))

                    model_state = {
                        'epoch': pre_epoch,
                        'state_dict': self.lobe.state_dict(),
                        'vlu_dict': self.vlu.state_dict(),
                        'vls_dict': self.vls.state_dict(),
                    }
                    early_stopping(mean_score.item(), model_state)

                    if early_stopping.is_best:
                        best_model = {
                            'epoch': pre_epoch,
                            'lobe': self.lobe.state_dict(),
                            'vlu': self.vlu.state_dict(),
                            'vls': self.vls.state_dict()
                        }

                    if early_stopping.early_stop:
                        print("Early stopping triggered for full network training")
                        break

                    if epoch % 10 == 9:
                        if self.args.save_checkpoints:
                            torch.save(model_state, f"{self.args.save_folder}/logs_{epoch}.pt")

                        train_mean = np.mean(self.vampnet.train_scores[-batch_count - 1:-1][0, 1])
                        record_result(
                            f"Epoch: {epoch}, "
                            f"validation score: {mean_score.item():.4f}, "
                            f"training mean: {train_mean:.4f}, "
                            f"best score: {early_stopping.val_loss_min:.4f}",
                            self.log_pth
                        )

            except RuntimeError as e:
                print(f"Runtime error at epoch {epoch}:")
                print(e, e.args[0])
                n_OOM += 1

        if best_model:
            print(f"Loading best model (score: {early_stopping.val_loss_min:.4f})")
            self.vampnet.lobe.load_state_dict(best_model['lobe'])
            self.vampnet.lobe_timelagged.load_state_dict(deepcopy(best_model['lobe']))
            self.vampnet.vampu.load_state_dict(best_model['vlu'])
            self.vampnet.vamps.load_state_dict(best_model['vls'])

        return self.vampnet.fetch_model(), all_train_epo

    def save_training_metrics(self, model):
        """Save training and validation scores."""
        with open(f'{self.args.save_folder}/train_scores.npy', 'wb') as f:
            np.save(f, model.train_scores)

        with open(f'{self.args.save_folder}/validation_scores.npy', 'wb') as f:
            np.save(f, model.validation_scores)

    def train(self):
        """Execute the complete training pipeline."""
        print("Starting model training...")
        model, all_train_epoch = self.train_vamp()
        self.save_training_metrics(model)
        print("Model training completed")
        return model, all_train_epoch


def main():
    args = buildParser().parse_args()
    # Flush cache before processing each batch
    flush_cuda_cache()
    trainer = RevVAMPTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
