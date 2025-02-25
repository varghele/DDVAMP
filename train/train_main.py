import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from args import buildParser
from utils.unflatten import unflatten
from utils.count_parameters import count_parameters

from deeptime.util.data import TrajectoryDataset
from components.models.GraphVAMPNet import GraphVampNet
from components.models.RevVAMPNet import RevVAMPNet
from train.tr_utils.EarlyStopping import EarlyStopping
from tqdm import tqdm
from typing import Optional, Tuple
from components.models.vamps import VAMPS
from components.models.vampu import VAMPU
import numpy as np
import random
from components.activations.ExpActivation import ExpActivation
import matplotlib.pyplot as plt
from utils.vamp_utils import *

# Removing stochasticity, setting seed so training is always the same
#torch.backends.cudnn.deterministic = True
#random.seed(hash("setting random seeds") % 2**32 - 1)
#np.random.seed(hash("improves reproducibility") % 2**32 - 1)
#torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
#torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def flush_cuda_cache():
    """Flush CUDA cache and clear memory."""
    try:
        # Empty CUDA cache
        torch.cuda.empty_cache()

        # Synchronize CUDA streams
        torch.cuda.synchronize()

    except Exception as e:
        print(f"Warning: Could not flush CUDA cache: {str(e)}")

def record_result(arg1, arg2):
    pass

class RevVAMPTrainer:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()

        # Get project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Infer parameters from input directory path
        self.inferred_params = self._infer_params_from_path(args.data_path)
        self.setup_directories()

        # Initialize data
        self.traj_length, self.dists1, self.inds1 = self._load_data()
        self.dataset = self._prepare_dataset()
        self.loader_train, self.loader_val, self.loader_train_all = self._create_dataloaders()

        # Initialize model
        self.lobe, self.vampnet, self.vlu, self.vls = self._initialize_model()

        self.all_train_epoch = 0

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
        """Set up logging path."""
        self.log_pth = os.path.join(self.args.save_folder, 'training.log')

        # Save arguments as text
        args_file = os.path.join(self.args.save_folder, 'meta.txt')
        with open(args_file, 'w') as f:
            f.write(str(self.args))
            f.write("\n\nInferred parameters:\n")
            f.write(str(self.inferred_params))

    def _infer_params_from_path(self, data_path):
        """Infer parameters from the data directory path."""
        # Expected format: data/proteinname/interim/proteinname_numbernbrs_numberns/
        try:
            dir_name = os.path.basename(os.path.normpath(data_path))
            parts = dir_name.split('_')

            params = {
                'protein_name': parts[0],
                'num_neighbors': int(parts[1].replace('nbrs', '')),
                'ns': int(parts[2].replace('ns', ''))
            }

            print(f"Inferred parameters from path:")
            print(f"Protein name: {params['protein_name']}")
            print(f"Number of neighbors: {params['num_neighbors']}")
            print(f"Nanoseconds: {params['ns']}")

            return params

        except Exception as e:
            raise ValueError(f"Failed to infer parameters from path: {data_path}. "
                             f"Expected format: data/proteinname/interim/proteinname_numbernbrs_numberns/")

    def _load_data(self):
        """Load and prepare data from files."""
        try:
            # Load data from the inferred directory
            dist_file = os.path.join(self.args.data_path, "dist_min.npy")
            inds_file = os.path.join(self.args.data_path, "inds_min.npy")

            dists1 = np.load(dist_file)
            inds1 = np.load(inds_file)

            # Infer trajectory length from data shape
            traj_length = [dists1.shape[0]]

            return traj_length, dists1, inds1

        except Exception as e:
            raise ValueError(f"Failed to load data files from {self.args.data_path}: {str(e)}")

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
        lobe = GraphVampNet(args=self.args)
        lobe_timelagged = deepcopy(lobe).to(device=self.device)
        lobe = lobe.to(self.device)

        print('Number of parameters:', count_parameters(lobe))

        vlu = VAMPU(self.args.num_classes, activation=ExpActivation())
        vls = VAMPS(self.args.num_classes, activation=ExpActivation(), renorm=True)

        vampnet = RevVAMPNet(
            lobe=lobe,
            lobe_timelagged=lobe_timelagged,
            vampu=vlu,
            vamps=vls,
            learning_rate=self.args.learning_rate_a,
            device=self.device,
            activation_vamps=ExpActivation(),
            activation_vampu=ExpActivation(),
            num_classes=self.args.num_classes,
            tau=self.args.tau,
            optimizer='Adam',
            score_method=self.args.score_method,
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
        data_size = 0
        # First calculate the exact size needed by summing up all batch sizes
        with torch.no_grad():
            for batch_0, batch_t in self.loader_train:
                data_size += batch_0.shape[0]
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

        # TODO: Why?
        self.vampnet.update_auxiliary_weights([state_probs, state_probs_tau], optimize_S=True)

        # Train auxiliary networks
        best_dict = None
        for epoch in tqdm(range(n_epochs)):
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

    def train_vamp(self) -> Tuple[RevVAMPNet, int]:
        """Train the VAMPNet model with optional VAMPCE pre-training."""
        n_OOM = 0
        pre_epoch = self.args.pre_train_epoch

        if self.args.score_method == 'VAMPCE':
            print("Training vanilla VAMPNet model...")
            self.vampnet.check_gradients()
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
            print("Entering Pre-Training: Training CHI")

            # Set learning rate to learning rate A
            self.vampnet.set_optimizer_lr(self.args.learning_rate_a)

            for epoch in tqdm(range(pre_epoch)):
                try:
                    for batch_0, batch_t in self.loader_train:
                        torch.cuda.empty_cache()
                        b_0 = batch_0.to(self.device)
                        b_t = batch_t.to(self.device)
                        self.vampnet.partial_fit((b_0, b_t))
                        torch.cuda.synchronize()

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

        # Establish learning rate A for auxiliary networks
        self.vampnet.set_optimizer_lr(self.args.learning_rate_a)

        # Train auxiliary networks
        self.train_US(pre_epoch)

        # Train full network
        print("Training full network...")
        # Set learning rate to learning rate B for full network training
        self.vampnet.set_optimizer_lr(self.args.learning_rate_b)

        all_train_epo = 0
        best_model = None
        early_stopping = EarlyStopping(
            self.args.save_folder,
            file_name='best_allnet',
            delta=1e-4,
            patience=200
        )

        for epoch in tqdm(range(self.args.epochs)):
            # Reduce larning rate to converge training
            if epoch == 100:
                self.vampnet.reduce_optimizer_lr(0.2)

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
        with open(os.path.join(self.args.save_folder, 'train_scores.npy'), 'wb') as f:
            np.save(f, model._train_scores)

        with open(os.path.join(self.args.save_folder, 'validation_scores.npy'), 'wb') as f:
            np.save(f, model._validation_scores)

            # Create visualization
            plt.figure()
            plt.loglog(*model._train_scores[-self.all_train_epoch:].T, label='training')
            plt.loglog(*model._validation_scores.T, label='validation')
            plt.xlabel('step')
            plt.ylabel('score')
            plt.legend()
            plt.savefig(os.path.join(self.args.save_folder, 'scores.png'))

    def train(self):
        """Execute the complete training pipeline."""
        print("Starting model training...")
        model, all_train_epoch = self.train_vamp()
        self.all_train_epoch = all_train_epoch
        self.save_training_metrics(model)
        print("Model training completed")
        return model, all_train_epoch

# For pipeline
def run_training(args):
    """
    Entry point for pipeline integration with analysis capabilities.
    """
    # Clear CUDA cache before training
    flush_cuda_cache()

    # Initialize and train model
    trainer = RevVAMPTrainer(args)
    model, all_train_epoch = trainer.train()

    # Prepare data for analysis
    data_np = [traj.cpu().numpy() for traj in trainer.dataset.trajectories]

    # Run model output analysis and get results directly
    probs, embeddings, attentions = analyze_model_outputs(
        model=model,
        data_np=data_np,
        save_folder=args.save_folder,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        h_g=args.h_g,
        num_atoms=args.num_atoms,
        num_neighbors=args.num_neighbors
    )

    # Calculate and plot transition probabilities
    plot_transition_probabilities(
        probs=probs,
        save_dir=args.save_folder,
        protein_name=args.protein_name,
        lag_time=args.tau #20 #TODO: args.tau  # You can adjust this parameter or add it to args
    )

    # Use the indices already loaded in trainer
    neighbor_indices = [trainer.inds1]

    # Run Chapman-Kolmogorov test
    steps = 10
    tau_msm = args.tau #20 #TODO: this is args.tau
    predicted, estimated = get_ck_test(probs, steps, tau_msm)

    # Plot and save CK test results
    plot_ck_test(predicted, estimated, args.num_classes, steps, tau_msm, args.save_folder)
    np.savez(os.path.join(args.save_folder, 'ck.npz'), list((predicted, estimated)))
    print("CK test complete")

    # Calculate and plot implied timescales
    max_tau = 250
    lags = [i for i in range(1, max_tau, 2)]
    its = get_its(probs, lags)
    # Using save_path instead of save_folder to match the function signature
    plot_its(its, lags, save_path=args.save_folder, ylog=False)
    np.save(os.path.join(args.save_folder, 'ITS.npy'), np.array(its))
    print("ITS calculation complete")

    # Process attention maps
    # Get state assignments
    state_assignments = [np.argmax(traj, axis=1) for traj in probs]

    # Calculate attention maps with pre-calculated neighbor indices
    state_attention_maps, state_populations = calculate_state_attention_maps(
        attentions=attentions,
        neighbor_indices=neighbor_indices,
        state_assignments=state_assignments,
        num_classes=args.num_classes,
        num_atoms=args.num_atoms
    )

    # Save state assignments and attention maps and state_populations
    save_state_data(
        state_assignments=state_assignments,
        state_attention_maps=state_attention_maps,
        state_populations=state_populations,
        save_path=args.save_folder,
        protein_name=args.protein_name
    )

    # Plot attention maps
    plot_state_attention_maps(
        adjs=state_attention_maps,
        states_order=np.argsort(state_populations)[::-1],
        n_states=args.num_classes,
        state_populations=state_populations,  # Add this parameter
        save_path=os.path.join(args.save_folder, 'attention_maps.png')
    )

    # Plot attention weights
    plot_state_attention_weights(
        state_attention_maps=state_attention_maps,
        topology_file=os.path.join(os.path.dirname(args.data_path), f"{args.protein_name}.pdb"),
        n_states=args.num_classes,
        save_path=os.path.join(args.save_folder, f'{args.protein_name}_attention_weights.png')
    )
    print("Attention analysis complete")

    # Get state structures
    print("Generating state structures")
    # After getting state assignments
    state_structures = generate_state_structures(
        traj_folder=args.traj_folder,
        topology_file=os.path.join(os.path.dirname(args.data_path), f"{args.protein_name}.pdb"),
        state_assignments=state_assignments,
        save_dir=args.save_folder,
        protein_name=args.protein_name,
        stride=1,#args.stride,  # TODO: Adjust this value to maybe get into ns range 100-1000 a good start
        # TODO: This needs to be a separate argument than stride
        n_structures=10
    )

    visualize_state_ensemble(
        state_structures=state_structures,
        save_dir=args.save_folder,
        protein_name=args.protein_name
    )
    print("Representative state structures generated")

    # Generate state structures with attention coloring
    # Visualize existing structures with attention coloring
    visualize_attention_ensemble(
        state_structures=state_structures,
        state_attention_maps=state_attention_maps,
        save_path=os.path.join(args.save_folder, f"{args.protein_name}_attention.pdb"),
        protein_name=args.protein_name
    )
    print("Attention-colored visualizations generated")

    # After analyzing model outputs and generating state structures
    plot_state_network(
        probs=probs,
        state_structures=state_structures,  # From generate_state_structures
        save_dir=args.save_folder,
        protein_name=args.protein_name,
        lag_time=args.tau
    )
    print("network of states plotted ")


    return {
        'model': model,
        'epochs_trained': all_train_epoch,
        'probabilities': probs,
        'embeddings': embeddings,
        'attentions': attentions,
        'predictions': predicted,
        'estimations': estimated,
        'implied_timescales': its,
        'lags': lags,
        'state_populations': state_populations,
        'state_attention_maps': state_attention_maps,
        'state_structures': state_structures
    }



def main():
    args = buildParser().parse_args()
    # Flush cache before processing each batch
    flush_cuda_cache()
    trainer = RevVAMPTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
