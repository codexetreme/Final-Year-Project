import torch
import nonechucks as nc
import torch.utils.data as datautil

from utils import *
from DQN import *
from reward import Reward
from config import Configuration
from data_loader import TextDataset

import os
from tqdm import tqdm

def test():

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

	configuration = Configuration('project.config')
	config = configuration.get_config_options()

	# create_vocabulary_from_dataset(config)	
	# make_target_vocab(config)
	
	word2idx,dataset_vectors = load_target_vocab(config)
	
	policy_net.load_state_dict(torch.load(config.paths.PATH_TO_WEIGHTS))
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	def select_action(config,doc,state):
		"""
		doc contains sentences
		"""
		sample = np.random.random()
		# Putting this here as we need q_values one way or the other
		q_values = target_net(doc,get_q_approx=True,sum_i=state['sum_i'])

		if iter%1000 == 0:
			config.dqn.EPS_START -= config.dqn.EPS_DECAY
		if sample < config.dqn.EPS_START:
			i = np.random.randint(low=0,high=doc.shape[1]-1)
			a_i = (i,doc[0,i])
		else:
			# sentence representation are calculated as no grad because we dont want to disturb / update the weights 
			# when calculating sentence represntations only the Q^{i}_{j}(s,a) function needs update, as the sencence 
			# and document encoders are updated later in the code. 
			
			# actions are sentences
			# TODO: check if we really need no_grad()
			# with torch.no_grad():
			# q_values = policy_net(doc,get_q_approx=True,sum_i=state['sum_i'])

			i = torch.argmax(q_values)
			a_i = (i,doc[0,i])
		return a_i,q_values

	doc_summaries = []
	for i,(story,highlights,text) in tqdm(enumerate(data_loader)):
		story = story.to(device)
		highlights = highlights.to(device)
		state = {'curr_summary_ids':[],'curr_summary':[],'sum_i':torch.zeros((100))}
		for j in config.evals.MAX_ITERS:
			H_i,D_i,x = target_net(story['tensor'][:,:len(story['raw'])-1])
			a_i,q_values = select_action(config,story,state)
			state['curr_summary_ids'].append(int(a_i[0]))
			state['curr_summary'].append(a_i[1])
		doc_summaries.append(state)		
			
			

def test_logic():
	import torch
	# a = torch.empty(2).uniform_(1, 3)
	a = torch.randint(1,5,size=(2,))
	b = torch.randint(1,5,size=(2,3))
	k = torch.matmul(a,b)
	print (k)
	print (a)
	print (b)


if __name__ == '__main__':
	test()
	# test_logic()
	# worker()