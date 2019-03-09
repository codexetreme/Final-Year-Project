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

def main():

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

	configuration = Configuration('project.config')
	config = configuration.get_config_options()

	# create_vocabulary_from_dataset(config)	
	# make_target_vocab(config)
	
	word2idx,dataset_vectors = load_target_vocab(config)

	use_safe_dataset = True
	dataset = TextDataset(word2idx,dataset_vectors,config=config)
	if use_safe_dataset:
		dataset = nc.SafeDataset(dataset)
		data_loader = nc.SafeDataLoader(dataset=dataset,batch_size=config.globals.BATCH_SIZE,num_workers=0,shuffle=True)
	else:
		data_loader = datautil.DataLoader(dataset=dataset,batch_size=config.globals.BATCH_SIZE,num_workers=0,shuffle=False)

	# model = SentenceEncoder(target_vocab = word2idx.keys(), vectors = dataset_vectors, config = config)
	# doc_enc = DocumentEncoder(config=config)
	
	policy_net = DQN(target_vocab = word2idx.keys(),vectors = dataset_vectors,config = config).to(device)
	target_net = DQN(target_vocab = word2idx.keys(),vectors = dataset_vectors,config = config).to(device)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()
	
	reward_func = Reward()
	h = reward_func.get_reward([['hello']],[[['this is a good hello']]])
	print (h)

	def select_action(config,doc,state):
		# TODO: fix the function to handle the full batchsize
		# TODO: send all tensors to GPU

		sample = np.random.random()

		# article = '\n'.join(doc['raw'])		
		# article = article.split('\n\n')
		doc_tensor = doc['tensor'][:,:len(doc['raw'])-1]
		# Putting this here as we need q_values one way or the other
		q_values = policy_net(doc_tensor,get_q_approx=True,sum_i=state['sum_i'])

		# Decay the epsilon per EPS_DECAY_ITER iterations
		if iter % config.dqn.EPS_DECAY_ITER == 0:
			config.dqn.EPS_START -= config.dqn.EPS_DECAY
			print (config.dqn.EPS_START)

		if sample < config.dqn.EPS_START:
			i = np.random.randint(low=0,high=len(doc['raw'])-1)
		else:
			# actions are sentences
			i = torch.argmax(q_values,dim=1)

		a_i = (i,doc['raw'][i])
		return a_i,q_values

	optimizer = torch.optim.RMSprop(policy_net.parameters())
	memory = ReplayMemory(config.dqn.REPLAY_MEM_SIZE)
	
	epoch = 0
	iter = 0
	
	for epoch in tqdm(range(epoch,config.globals.NUM_EPOCHS)):
		policy_net.train()
		for i,(story,highlights) in tqdm(enumerate(data_loader)):

			state = {'curr_summary_ids':[],'curr_summary':[],'sum_i':torch.zeros((100))}
			iter = iter + 1

			# if i>20 : break

			story['tensor'] = story['tensor'].to(device)
			highlights['tensor'] = highlights['tensor'].to(device)
			# sentence representation are calculated as no grad because we dont want to disturb / update the weights 
			with torch.no_grad():
				H_i,D_i,x = policy_net(story['tensor'][:,:len(story['raw'])-1])

			a_i,q_values = select_action(config,story,state)
			
			state['curr_summary_ids'].append(int(a_i[0]))
			state['curr_summary'].append(a_i[1])
			state['sum_i'] = Sum_i(H_i,state['curr_summary_ids'],q_values)
			r_i = reward_func.get_reward([state['curr_summary']],gold_summ=[[highlights['raw']]])
			
			

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
	main()
	# test_logic()
	# worker()