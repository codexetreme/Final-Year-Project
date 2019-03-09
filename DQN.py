import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import SentenceEncoder,DocumentEncoder

from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)





def Sum_i(h_values,h_idx,q_values):
	"""
	numerator is the q_values * H_I_J of the sentences in the summary
	denominator is the sum of q_values of all sentences in the doc
	"""
	
	denom = torch.sum(q_values,dim=1) # [1]
	# this vecotrizes the summation operation
	sum = torch.matmul(q_values[:,h_idx],h_values[:,h_idx]).squeeze() # [100]
	return sum/denom # [100]


class Q_i_j(nn.Module):
	'''
	This module calculates the Q^{i}_{j} approximation function.

	The salience, content, novelty and the posistion embeddings are calculated as shown in nallapati et al, SummaRuNNer

	Positional Embeddings are calculated as follows:
		1> absoulute position = this is the absoulute position of the sentence in the document
		2> relative position = this is the relative position in the document, this is calculated by dividing the document 
			into a set of segments and then the sentence's location is found by the following formula:
				rel_pos = (abs_position + 1) * (num_seg-1) / document_length
	
	NOTE: Lookup Nallapati et al, SummaRuNNer's paper for a mathematical explaination of this relative position.

	Once these values are calulated, we make an embedding for these and calulate the absoulute and relative embedding.
	When added, these give the positional embedding.

	'''
	def __init__(self,config,bidirectional=True,hidden_size=50):
		"""
		Setup the Q_i_j function's parameters.
		
		Args:
			config: an instance of the Config class
				bidirectional: Indicate if the RNN that was used to calulate the sentence hidden states was bidirectional 
				default = True
			
			hidden_size: Size of the hidden representation of the sentences
				default = 50 

			use_relative_embedding: If relative positional embedding is to be calculated, set to true.
				default = True

		"""   
		super(Q_i_j,self).__init__()
		self.hidden_size = hidden_size
		self.config = config
		if bidirectional:
		  self.hidden_size *= 2
		
		# FIXME: these values, specially for pos_num may not work
		pos_dim = self.config.q_func.POS_DIM
		pos_num = self.config.q_func.POS_NUM
		seg_num = self.config.q_func.SEG_NUM

		self.config = config
		
		self.content = nn.Linear(self.hidden_size,1,bias=False)
		self.salience = nn.Bilinear(self.hidden_size,self.hidden_size,1,bias=False)
		self.redundancy = nn.Bilinear(self.hidden_size,self.hidden_size,1,bias=False)
		
		self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))
		
		self.abs_pos = nn.Linear(pos_dim,1,bias=False)
		self.abs_pos_embed = nn.Embedding(pos_num,pos_dim)

		if self.config.q_func.USE_RELATIVE_EMBEDDING:
			self.rel_pos_embed = nn.Embedding(seg_num,pos_dim)
			self.rel_pos = nn.Linear(pos_dim,1,bias=False)

	def forward(self,h,h_idx,doc_len,D_i,sum_i):
		"""
		Forward pass of the Q_i_j approximation function
		
		Args:
			h: The hidden state representation of a sentence in the document 
			h_idx: Absoulute position of the sentence in the document
			doc_len: Document Length (AKA no. of sentences in the document)
			D_i: Document Score of the i_th document
			multiply_h=False: If you want to multiply the hidden vector H along with the Q function

		Returns:
			float : the q_value of the action
		"""   
		# TODO: this function can be vectorized
		abs_features = self.abs_pos_embed(torch.LongTensor([[h_idx]])).squeeze(0)
		content = self.content(h)
		salience = self.salience(h,D_i)
		redundancy = -1 * self.redundancy(h,torch.tanh(sum_i))
		abs_p = self.abs_pos(abs_features)
		q_value = content + salience + redundancy + abs_p + self.bias
		if self.config.q_func.USE_RELATIVE_EMBEDDING: 
			h_rel_id = int(round((h_idx + 1) * 9.0 / doc_len))
			rel_features = self.rel_pos_embed(torch.LongTensor([[h_rel_id]])).squeeze(0)
			rel_p = self.rel_pos(rel_features)
			q_value = q_value + rel_p
		return q_value



class DQN(nn.Module):

	def __init__(self,target_vocab,vectors,config):
		super(DQN,self).__init__()
		self.SentenceEnc = SentenceEncoder(target_vocab,vectors,config=config)
		self.DocumentEnc = DocumentEncoder(config=config)
		self.Q_func = Q_i_j(config)
		self.config = config
	# Parameters of Classification Layer
	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x,get_q_approx=False,sum_i=0):
		"""
		
		Returns:
			H_i (tensor): H_i, the sentence representation
			D_i (tensor): D_i document score 
		"""
		x = self.SentenceEnc(x)
		D_i,H_i = self.DocumentEnc(x)
		# print ('H_i.shape',H_i.shape)
		# print ('D_i.shape',D_i.shape)

		if get_q_approx:
			q_values = []
			for _ in range(self.config.globals.BATCH_SIZE):
				sentences = H_i[_]
				for i,h in enumerate(sentences):
					# H_i => all actions possible
					q = self.Q_func(h,h_idx=i,doc_len=len(sentences),D_i=D_i[_],sum_i=sum_i)
					q_values.append(q)
					pass
			return torch.Tensor(q_values).view(self.config.globals.BATCH_SIZE,-1)
		return H_i,D_i,x


# Parameters of Classification Layer
