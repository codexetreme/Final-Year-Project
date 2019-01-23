import torch
import numpy as np
import torch.nn as nn

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

		self.conv1 = nn.Conv1d(in_channels=1,out_channels=40,kernel_size=(1,self.config.WORD_DIMENTIONS))
		self.conv2 = nn.Conv1d(in_channels=1,out_channels=50,kernel_size=(2,self.config.WORD_DIMENTIONS))
		self.conv3 = nn.Conv1d(in_channels=1,out_channels=50,kernel_size=3)
		self.conv4 = nn.Conv1d(in_channels=1,out_channels=60,kernel_size=4)
		self.conv5 = nn.Conv1d(in_channels=1,out_channels=60,kernel_size=5)
		self.conv6 = nn.Conv1d(in_channels=1,out_channels=70,kernel_size=6)
		self.conv7 = nn.Conv1d(in_channels=1,out_channels=70,kernel_size=7)
	
		self.max_pool = nn.MaxPool1d(kernel_size=(1,self.config.WORD_DIMENTIONS))


	def forward(self,x):
		x = self.embedding_layer(x)
		x = x.unsqueeze(1).view(x.shape[0],1,-1,self.config.WORD_DIMENTIONS)

		# print ("x: ",x.shape)
		x_1 = self.conv1(x)
		# x_2 = self.conv2(x)
		# x_1 = self.max_pool(x_1)
		# print(x_1.shape)
		# x_2 = self.conv2(x)
		# x_3 = self.conv3(x)
		# x_4 = self.conv4(x)
		# x_5 = self.conv5(x)
		# x_6 = self.conv6(x)
		# x_7 = self.conv7(x)

		return x_1








