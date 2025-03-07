# Author: Mahdi Ghorbani

import torch
import numpy as np
import deeptime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from sklearn.neighbors import BallTree
from args import buildParser
from layers import GaussianDistance, GraphConvLayer, NeighborMultiHeadAttention, LinearLayer, ContinuousFilterConv, InteractionBlock
from layers import GATLayer
from layers import GraphAttentionLayer
#from torch_scatter import scatter_mean
import time

#from torch_scatter import scatter_mean

args = buildParser().parse_args()

if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')

class StandVampNet(nn.Module):
	"""
	"""
	def __init__(self, n_in = [30,22,16,12,9], n_out=6, dropout=[], weight_init='xavier'):
		super(StandVampNet, self).__init__()
		self.nlayer = n_in
		self.n_out = n_out
		self.layer = nn.ModuleList()
		for i in range(1, len(self.nlayer)):
			self.layer.append(nn.Linear(self.nlayer[i-1], self.nlayer[i], bias=True))
			if len(dropout) > i and dropout[i] > 0.0:
				self.layer.append(nn.Dropout(dropout[i]))
		self.out_layer = nn.Linear(self.nlayer[-1], self.n_out, bias=True)

		# with torch.no_grad():
		# 	if weight_init == 'xavier':
		# 		torch.nn.init.xavier_uniform_(seq[0].weight)
		# 	if weight_init == 'identity':
		# 		torch.nn.init.eye_(seq[0].weight)
		# 	if weight_init not in ['xavier', 'identity', None]:
		# 		if isinstance(weight_init, int) or isinstance(weight_init, float):
		# 			torch.nn.init.constant_(seq[0].weight, weight_init)



	def forward(self, input):
		tmp_in = input
		for idx in range(len(self.layer)):
			tmp_out = self.layer[idx](tmp_in)
			tmp_in = F.selu(tmp_out)

		out = self.out_layer(tmp_out)
		out = F.softmax(out,  dim=-1)
		return out

class GraphVampNet(nn.Module):
	''' wrapper class for different types of graph convolutions: ['GraphConvLayer', 'NeighborMultiHeadAttention', 'SchNets']

	parameters:
	-----------------
	seq_file: text file,
			the sequence file for initializing a one-hot encoding for embedding of different atoms
			If not provided, a random initilizer will be used for atom embeddings
	num_atoms: int,
			number of atoms
	num_neighbors: int,
			number of neighbors of each atom to be considered.
	tau: int,
			lagtime to be considered for the model
	n_classes: int,
			number of classes to cluster the output graphs
	n_conv: int,
			number of convlutional layers on graphs
	dmin: float,
			the minimum distance for performing the gaussian basis expansion
	dmax: float,
			the maximum distance for performing the gaussian basis expansion
	step: float,
			the step for gaussian basis expansion
	h_a: int,
			number of dimensions for atom embeddings
	h_g: int,
			number of dimension for after pooling
	atom_embedding_init: str,
			the type of initialization for the atom embedding
	use_pre_trained: boolean,
			Whether to use a pretrained embedding for atoms
	activation:
			the non-linear activation function to be used in the model
	pre_trained_weights_file: str,
			the filename for pretrained embedding
	conv_type: the type of convlutional layer for graphs ['GraphConvLayer', 'NeighborMultiHeadAttention', 'SchNets']
	num_heads: int,
			the number of heads for multi-head attention in NeighborMultiHeadAttention convlution type
	residual: boolean,
			Whether to add a residual connection between different convolutions
	use_backbone_atoms: boolean,
			Whether to use backbone atoms for the protein graph model (if False will use the Ca atoms only)

	attention_pool: bool
			Whether to add Graph Attention layer between embedding learned before performing graph pooling
	'''
	def __init__(self,
				 seq_file=args.seq_file,
				 num_atoms=args.num_atoms,
				 num_neighbors=args.num_neighbors,
				 n_classes=args.num_classes,
				 n_conv=args.n_conv,
				 dmin=args.dmin,
				 dmax=args.dmax,
				 step=args.step,
				 h_a=args.h_a,
				 h_g=args.h_g,
				 atom_embedding_init='normal',
				 use_pre_trained=False, activation=nn.ReLU(),
				 pre_trained_weights_file=None,
				 conv_type=args.conv_type,
				 num_heads=args.num_heads,
				 residual=args.residual,
				 use_backbone_atoms=args.use_backbone_atoms,
				 dont_pool_backbone=args.dont_pool_backbone,
				 attention_pool=args.attention_pool):

		super(GraphVampNet, self).__init__()
		self.seq_file = seq_file
		self.num_atoms = num_atoms
		self.num_neighbors = num_neighbors
		self.n_classes= n_classes
		self.n_conv = n_conv
		self.dmin = dmin
		self.dmax = dmax
		self.step = step
		self.h_a = h_a
		self.h_g = h_g
		self.gauss = GaussianDistance(dmin, dmax, step)
		self.h_b = self.gauss.num_features # number of gaussians, #[B, N, M, bond_fea_len]
		self.num_heads = num_heads
		self.activation = nn.ReLU()
		self.use_backbone_atoms = use_backbone_atoms
		self.residual = residual
		#self.atom_emb = nn.Embedding(num_embeddings=self.num_atoms, embedding_dim=self.h_a)
		#self.atom_emb.weight.data.normal_()
		self.conv_type = conv_type
		if self.conv_type == 'GraphConvLayer':
			self.convs = nn.ModuleList([GraphConvLayer(self.h_a,
												 	   self.h_b) for _ in range(self.n_conv)])

		elif self.conv_type == 'NeighborMultiHeadAttention':
			self.convs = nn.ModuleList([NeighborMultiHeadAttention(self.h_a,
														  self.h_b,
														  self.num_heads) for _ in range(self.n_conv)])
		elif self.conv_type == 'GATLayer':
			self.convs = nn.ModuleList([GATLayer(self.h_a,
	                                                       self.h_b,
	                                                       self.num_heads) for _ in range(self.n_conv)])
		elif self.conv_type == 'SchNet':
			self.convs = nn.ModuleList([InteractionBlock(n_inputs=self.h_a,
														 n_gaussians=self.h_b,
														 n_filters=self.h_a,
														 activation=nn.Tanh()) for _ in range(self.n_conv)])

		self.conv_activation = nn.ReLU()
		if self.h_g is not None:
			self.fc_classes = nn.Linear(self.h_g, n_classes)
		else:
			self.fc_classes = nn.Linear(self.h_a, n_classes)
		self.init = atom_embedding_init
		self.use_pre_trained = use_pre_trained
		self.dont_pool_backbone = dont_pool_backbone
		self.attention_pool = attention_pool

		#elf.weight = nn.Parameter(torch.Tensor(self.h_a, 1))

		#if args.use_backbone_atoms:
		if args.h_g is not None:
			self.amino_emb =  nn.Linear(self.h_a, self.h_g)

		if use_pre_trained:
			self.pre_trained_emb(pre_trained_weights_file)

		elif seq_file is not None:
			atom_emb = self.onehot_encode_amino(seq_file)
			self.atom_embeddings = torch.tensor(atom_emb, dtype=torch.float32).to(device)
			self.h_init = atom_emb.shape[-1] # dimension of atom embedding [20]
			emb = nn.Embedding.from_pretrained(self.atom_embeddings, freeze=False)
			self.atom_emb = nn.Linear(self.h_init, self.h_a) # linear layer for atom features

		else:
			# initialize the atom embeddings randomly
			self.atom_emb = nn.Embedding(num_embeddings=self.num_atoms, embedding_dim=self.h_a)
			self.init_emb()

		if self.attention_pool:
			self.attn_pool_model = GATLayer(self.h_a, self.h_a, concat_heads=True, alpha=0.2)

	def pre_trained_emb(self, file):
		'''
		loads the pre-trained node embedings from a file
		For now we are not freezing the pre-trained embeddings since
		we are going to update it in the graph convolution
		'''

		with open(self.pre_trained_weights_file) as f:
			loaded_emb = json.load(f)

		embed_list = [torch.tensor(value, dtype=torch.float32) for value in loaded_emb.values()]
		self.atom_embeddings = torch.stack(embed_list, dim=0)
		self.h_init = self.atom_embeddings.shape[-1] # dimension atom embedding init
		self.atom_emb = nn.Embedding.from_pretrained(self, atom_embeddings, freeze=False)
		self.embedding = nn.Linear(self.h_init, self.h_a)

	def init_emb(self):
		'''
		Initialize random embedding for the atoms
		'''
		#--------------initialization for the embedding--------------
		if self.init == 'normal':
			self.atom_emb.weight.data.normal_()

		elif self.init == 'xavier_normal':
			self.atom_emb.weight.data._xavier_normal()

		elif self.init == 'uniform':
			self.atom_emb.weight.data._uniform()


	def onehot_encode_amino(self, seq_file):
		'''
		one-hot encoding of amino types for initializing the embedding
		'''
		with open(seq_file, 'r') as f:
			sequence = open(seq_file)
			seq = sequence.readlines()[0].strip()

		amino_dict = {'A':0,
					  'R':1,
					  'N':2,
					  'D':3,
					  'C':4,
					  'Q':5,
					  'E':6,
					  'G':7,
					  'H':8,
					  'I':9,
					  'L':10,
					  'K':11,
					  'M':12,
					  'F':13,
					  'P':14,
					  'S':15,
					  'T':16,
					  'W':17,
					  'Y':18,
					  'Z':19}

		if args.use_backbone_atoms:
			s_encoded = np.zeros((20, 3*len(seq)))
			for i, n in enumerate(s):
				s_encoded[amino_dict[n],i*3:i*3+3] = 1

		else:
			s_encoded = np.zeros((20, len(seq)))
			for i, n in enumerate(s):
				s_encoded[amino_dict[n],i] = 1

		return s_encoded.T

		#------------------------------------------------------------

	@staticmethod
	def convert_adj(mat):
		# convert the nbr_adj_list matrix to an adjacency matrix
		mat = mat.to(torch.long)
		adj = torch.zeros((mat.shape[0], mat.shape[1], mat.shape[1]))
		for i in range(mat.shape[0]):
			adj[i][torch.arange(adj.shape[1])[:,None],mat[i]] = 1
			adj[i] += torch.eye(mat.shape[1])
		return adj.to(device)

	def pooling(self, atom_emb):
		# global pooling layer by averaging the embedding of nodes to get embedding of graph

		summed = torch.sum(atom_emb, dim=1)
		return summed / self.num_atoms


	def pool_amino(self, atom_emb):
		'''
		pooling the features of atoms in each amino acid to get a feature vector for each residue
		parameters:
		--------------------------
		atom_emb: embedding of atoms [B,N,h_a]
		residue_atom_idx: mapping between every atom and every residue in the protein
				size: [N] example [0,0,0,1,1,1,2,2,2] for N=6 and NA=3

		Returns:
		--------------------------
		pooled features of amino acids in the graph
		[B, Na, h_a]
		'''

		B = atom_emb.shape[0]
		N = atom_emb.shape[1]
		h_a = atom_emb.shape[2]

		residue_atom_idx = torch.arange(N).repeat(1,3)
		residue_atom_idx = residue_atom_idx.view(3,N).T.reshape(-1,1).squeeze(-1)
		Na = torch.max(residue_atom_idx)+1 # number of residues
		pooled = scatter_mean(atom_emb, residue_atom_idx, out=atom_emb.new_zeros(B,Na,h_a), dim=1)
		return pooled

	def return_emb(self):
		'''
		returns the embedding learned for each amino acid (or Ca atom) after training the model
		'''
		return self.emb



	def forward(self, data, return_emb=False, return_attn=False):
		''' Graph neural net computation to get features of protein at each timestep of simulation

		data: has shape [batch-size, num_atoms, num_neighbors*2]
			the first half of last index contains the nbr_adj_dist -> distance between every two atoms
			the second half of data contains the the nbr_adj_list -> the index of neighbors of each atom

		This model:
			1. expands the nbr_adj_dist into gaussian basis function [batch-size, num-atoms, num-neighbors, n_gaussians]
			2. initalize the embedding and get the initial embedding of each atom
			3. perform graph convolution to propagate messages along nodes and edges multiple times
			4. pooling to get the features for the graph from node features
			5. Linear layer to get coarse grain for number of classes
			6. Apply a softmax activation for getting the class assignment probabilities

		'''
		M = data.shape[-1] # num_neighbors*2
		nbr_adj_dist = data[:,:,:M//2] # [batch-size, num_atoms, num_neighbors]
		nbr_adj_list = data[:,:,M//2:] # [batch-size, num_atoms, num_neighbors]
		N = nbr_adj_list.shape[1]
		B = nbr_adj_list.shape[0]

		nbr_emb = self.gauss.expand(nbr_adj_dist) # [batch-size, num_atoms, num_neighbors, n_gaussians]
		# this is the edge embedding

		atom_emb_idx = torch.arange(N).repeat(B,1).to(device)
		atom_emb = self.atom_emb(atom_emb_idx)
		# atom_emb [B,N,h_a]

		if args.conv_type == 'GraphConvLayer':
			for idx in range(self.n_conv):
				tmp_conv, attn_probs = self.convs[idx](atom_emb=atom_emb,
										   nbr_emb=nbr_emb,
										   nbr_adj_list=nbr_adj_list)
				if self.residual:
					atom_emb = atom_emb + tmp_conv
				else:
					atom_emb = tmp_conv

		elif args.conv_type == 'NeighborMultiHeadAttention':
			for idx in range(self.n_conv):
				tmp_conv, attn_probs = self.convs[idx](h_V=atom_emb,
										   h_E=nbr_emb,
										   mask_attend=nbr_adj_list)
				if self.residual:
					atom_emb = atom_emb + tmp_conv
				else:
					atom_emb = tmp_conv

		elif args.conv_type == 'GATLayer':
			for idx in range(self.n_conv):
				tmp_conv, attn_probs = self.convs[idx](h_V=atom_emb,
										   h_E=nbr_emb,
										   return_attn_probs=return_attn)
				if self.residual:
					atom_emb = atom_emb + tmp_conv
				else:
					atom_emb = tmp_conv

		elif args.conv_type == 'SchNet':
			for idx in range(self.n_conv):
				tmp_conv, attn_probs = self.convs[idx](features=atom_emb,
										   rbf_expansion=nbr_emb,
									       neighbor_list=nbr_adj_list)

				if self.residual:
					atom_emb = atom_emb + tmp_conv
				else:
					atom_emb = tmp_conv

		emb = self.conv_activation(atom_emb)
		# [batch-size, num-atoms, h_a]


		if args.use_backbone_atoms:
			# apply a pooling layer for backbone atoms to amino acid features
			emb = self.pool_amino(emb)

		#t1 = time.time()
		#adj = self.convert_adj(nbr_adj_list) # convert nbr_adj_list to an adjacency matrix
		#t2 = time.time()
		#print('time: ' + str(t2-t1))
		#if self.attention_pool:
		#	emb, attn_probs = self.attn_pool_model(emb, adj)


		#print(attn_probs.shape)

		# embedding for each atom or amino acid learned after training
		#print(emb.shape)
		self.emb = emb # [batch, N, h_a]


		# the last pooling layer for getting graph features
		#attn_logits = torch.matmul(self.emb, self.weight).reshape(B,N)
		#attn_probs = F.softmax(attn_logits, dim=-1)

		#h_a = self.emb.shape[2]
		#self.emb = self.emb.reshape((B,h_a,N))
		#self.prot_emb = torch.bmm(self.emb, attn_probs.unsqueeze(-1)).squeeze(-1)

		self.prot_emb = self.pooling(self.emb)
		if self.h_g is not None:
			self.prot_emb = self.amino_emb(self.prot_emb)
		# [B, h_a] or [B, h_g]
		#print(self.prot_emb.shape)
		self.class_logits = self.fc_classes(self.prot_emb)
		# [B, n_classes]
		self.class_probs = F.softmax(self.class_logits, dim=-1)
		if return_emb:
			return self.prot_emb, attn_probs
		else:
			return self.class_probs
