import argparse


def buildParser():
	"""Create and return an argument parser with all necessary parameters."""
	parser = argparse.ArgumentParser(description='Neural network parameters and training configuration')

	# Model Architecture
	model_group = parser.add_argument_group('Model Architecture')
	model_group.add_argument('--num_atoms', type=int, default=42, help='Number of atoms in the system')
	model_group.add_argument('--num_neighbors', type=int, default=10,
							 help='Number of neighbors for each atom in the graph')
	model_group.add_argument('--num_classes', type=int, default=6,
							 help='Number of output classes/coarse-grained states')
	model_group.add_argument('--n_conv', type=int, default=4, help='Number of convolution layers')
	model_group.add_argument('--h_a', type=int, default=16, help='Atom hidden embedding dimension')
	model_group.add_argument('--h_g', type=int, default=8, help='Embedding dimension after backbone pooling')
	model_group.add_argument('--hidden', type=int, default=16, help='Number of hidden neurons')
	model_group.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')

	# Distance Parameters
	dist_group = parser.add_argument_group('Distance Parameters')
	dist_group.add_argument('--dmin', type=float, default=0.0, help='Minimum distance for the gaussian filter')
	dist_group.add_argument('--dmax', type=float, default=3.0, help='Maximum distance for the gaussian filter')
	dist_group.add_argument('--step', type=float, default=0.2, help='Step size for the gaussian filter')

	# Model Configuration
	config_group = parser.add_argument_group('Model Configuration')
	config_group.add_argument('--conv_type', type=str, default='SchNet',
							  help='Convolution layer type: [GraphConvLayer, NeighborMultiHeadAttention, SchNet]')
	config_group.add_argument('--num_heads', type=int, default=2, help='Number of heads in multihead attention')
	config_group.add_argument('--residual', action='store_true', help='Use residual connections')
	config_group.add_argument('--attention_pool', action='store_true', help='Use attention pooling')
	config_group.add_argument('--atom_init', type=str, default='normal', help='Initial embedding type for atoms')

	# Training Parameters
	training_group = parser.add_argument_group('Training Parameters')
	training_group.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
	training_group.add_argument('--tau', type=int, default=1, help='Lag time for the model')
	training_group.add_argument('--batch_size', type=int, default=32, help='Training batch size')
	training_group.add_argument('--val_frac', type=float, default=0.2, help='Validation fraction')
	training_group.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
	training_group.add_argument('--pre-train-epoch', type=int, default=2, help='Number of pre-training epochs')
	training_group.add_argument('--seed', type=int, default=42, help='Random seed')
	training_group.add_argument('--score_method', type=str, default='VAMPCE', help='Scoring method for VAMPNet')

	# System Configuration
	sys_group = parser.add_argument_group('System Configuration')
	sys_group.add_argument('--no-cuda', action='store_true', help='Disable CUDA training')
	sys_group.add_argument('--save-folder', type=str, default='logs', help='Where to save the trained model')
	sys_group.add_argument('--save_checkpoints', action='store_true', help='Store checkpoints during training')

	# Data Configuration
	data_group = parser.add_argument_group('Data Configuration')
	data_group.add_argument('--data-path', type=str, default='../intermediate/ala_5nbrs_1ns_',
							help='Data file prefix')
	data_group.add_argument('--seq_file', type=str, default=None,
							help='Sequence file for one-hot encoding initialization')

	return parser
