import argparse

def buildParser():

	parser = argparse.ArgumentParser()
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
	parser.add_argument('--seed', type=int, default=42, help='random seed')
	parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
	parser.add_argument('--batch-size', type=int, default=5000, help='batch-size for training')
	parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate')
	parser.add_argument('--hidden', type=int, default=16, help='number of hidden neurons')
	parser.add_argument('--num-atoms', type=int, default=10, help='Number of atoms')
	parser.add_argument('--num-classes', type=int, default=6, help='number of coarse-grained classes')
	parser.add_argument('--save-folder', type=str, default='logs', help='Where to save the trained model')
	parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
	parser.add_argument('--atom_init', type=str, default=None, help='inital embedding for atoms file')
	parser.add_argument('--h_a', type=int, default=16, help='Atom hidden embedding dimension')
	parser.add_argument('--num_neighbors', type=int, default=5, help='Number of neighbors for each atom in the graph')
	parser.add_argument('--n_conv', type=int, default=4, help='Number of convolution layers')
	parser.add_argument('--save_checkpoints', default=True,  action='store_true', help='If True, stores checkpoints')
	parser.add_argument('--conv_type', default='', type=str, help='the type of convolution layer, one of \
				        [GraphConvLayer, NeighborMultiHeadAttention, SchNet]')
	parser.add_argument('--dmin', default=0., type=float, help='Minimum distance for the gaussian filter')
	parser.add_argument('--dmax', default=3., type=float, help='maximum distance for the gaussian filter')
	parser.add_argument('--step', default=0.2, type=float, help='step for the gaussian filter')
	parser.add_argument('--tau', default=1, type=int, help='lag time for the model')
	parser.add_argument('--val-frac', default=0.3, type=float, help='fraction of dataset for validation')
	parser.add_argument('--num_heads', default=2, type=int, help='number of heads in multihead attention')
	parser.add_argument('--trained-model', default=None, type=str, help='path to the trained model for loading')
	parser.add_argument('--train', default=False, action='store_true', help='Whether to train the model or not')
	parser.add_argument('--use_backbone_atoms', default=False, action='store_true', help='Whether to use all the back bone atoms for training')
	parser.add_argument('--dont-pool-backbone', default=False, action='store_true', help='Whether not to pool backbone atoms')
	parser.add_argument('--h_g', type=int, default=8, help='Number of embedding dimension after backbone pooling')
	parser.add_argument('--seq_file', type=str, default=None, help='Sequence file to initialize a one-hot encoding based on amino types')
	parser.add_argument('--dist-data', type=str, default='dists_BBA_7nbrs_1ns.npz', help='the distnace data file')
	parser.add_argument('--nbr-data', type=str, default='inds_BBA_7nbrs_1ns.npz', help='the neighbors data file')
	parser.add_argument('--score-method', type=str, default='VAMP2', help='the scoring method of VAMPNet')
	parser.add_argument('--residual', action='store_true', default=False, help='Whether to use residual connections')
	parser.add_argument('--attention-pool', action='store_true', default=False, help= 'Whether to perform attention before global pooling')
	parser.add_argument('--return-emb', action='store_true', default=False, help='Whether return the learned graph embeddings')
	parser.add_argument('--return-attn', action='store_true', default=False, help='Whether to return the attention probs (only for NeighborMultiHeadAttention)')
	return parser
