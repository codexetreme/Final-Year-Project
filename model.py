import torch
import numpy as np
import torch.nn as nn

from utils import split_story, make_target_vocab, load_glove_vectors


def create_emb_layer(weights_matrix, trainable=True):
	num_embeddings, embedding_dim = weights_matrix.shape
	emb_layer = nn.Embedding(num_embeddings, embedding_dim)
	emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
	if not trainable:
		emb_layer.weight.requires_grad = False

	return emb_layer, num_embeddings, embedding_dim


class SentenceEncoder(nn.Module):
	def __init__(self,config=None):
		super(SentenceEncoder,self).__init__()
		
		if config is not None:
			self.config = config
		self.embedding_weights_matrix = create_embedding_matrix(self.config)
		self.embedding_layer,self.num_embeddings,self.embedding_dim = create_emb_layer(self.embedding_weights_matrix)

		print ("self.embedding_layer",self.embedding_layer)
		print ("self.num_embeddings",self.num_embeddings)
		print ("self.embedding_dim",self.embedding_dim)


	


	def forward(self):
		return 0








