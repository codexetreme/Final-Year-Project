import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import create_embedding_matrix


def create_emb_layer(weights_matrix, trainable=True):
	num_embeddings, embedding_dim = weights_matrix.shape
	emb_layer = nn.Embedding(num_embeddings, embedding_dim)
	emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
	if not trainable:
		emb_layer.weight.requires_grad = False

	return emb_layer, num_embeddings, embedding_dim


class SentenceEncoder(nn.Module):
	def __init__(self,target_vocab,vectors,config=None):
		super(SentenceEncoder,self).__init__()
		
		if config is not None:
			self.config = config
		self.embedding_weights_matrix = create_embedding_matrix(self.config,target_vocab,vectors)
		self.embedding_layer,self.num_embeddings,self.embedding_dim = create_emb_layer(self.embedding_weights_matrix)

		print ("self.embedding_layer",self.embedding_layer)
		print ("self.num_embeddings",self.num_embeddings)
		print ("self.embedding_dim",self.embedding_dim)

		Ks = [(1,40),(2,50),(3,50),(4,60),(5,60),(6,70),(7,70)]
		
		self.convs = nn.ModuleList([nn.Conv1d(1,out_channels=out_c,kernel_size=(k,self.config.WORD_DIMENTIONS)) for (k,out_c) in Ks])
	
		self.max_pool = nn.MaxPool1d(kernel_size=self.config.MAX_SENTENCES_PER_DOCUMENT)

	def conv_and_pool(self, x, conv):
		x = torch.tanh(conv(x)).squeeze(3)  # (N * doc_len, Co, W)
		x = F.max_pool1d(x, x.size(2)).squeeze(2) # (N * doc_len, Co)
		return x

	def forward(self,x):
		x = self.embedding_layer(x) # (N,doc_len,sentence_len,dims)
		x = x.unsqueeze(1).view(-1,1,self.config.MAX_SENTENCES_PER_DOCUMENT,self.config.WORD_DIMENTIONS) # (N*doc_len,Ci,sentence_len,dims)

		x = [self.conv_and_pool(x,i) for i in self.convs]
		x = torch.cat(x,dim=1)
		print ('x_shape:', x.shape)
			
		return x








