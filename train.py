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

	dataset = TextDataset(word2idx,dataset_vectors,config=config)
	# dataset = nc.SafeDataset(dataset)
	data_loader = datautil.DataLoader(dataset=dataset,batch_size=config.globals.BATCH_SIZE,num_workers=4,shuffle=False)

	# data_loader = nc.SafeDataLoader(dataset=dataset,batch_size=config.globals.BATCH_SIZE,num_workers=1,shuffle=True)
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
		"""
		doc contains sentences
		"""
		sample = np.random.random()
		# Putting this here as we need q_values one way or the other
		q_values = policy_net(doc,get_q_approx=True,sum_i=state['sum_i'])

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

	optimizer = torch.optim.RMSprop(policy_net.parameters())
	memory = ReplayMemory(config.dqn.REPLAY_MEM_SIZE)
	
	__i = 3
	epoch = 0
	iter = 0
	
	for epoch in tqdm(range(epoch,config.globals.NUM_EPOCHS)):
		policy_net.train()
		state = {'curr_summary_ids':[],'curr_summary':[],'sum_i':torch.zeros((100))}
		for i,(story,highlights,text) in tqdm(enumerate(data_loader)):
			iter = iter + 1

			if i>3:
				break
			
			story = story.to(device)
			highlights = highlights.to(device)
			print (text)
			
			# # Hidden representations for the document's sentences
			# # Squeezed, cause batch_size is 1
			# H_i,D_i,x = policy_net(story)
			# H_i.squeeze_(0)
			# print (np.array(text).shape)
			# print (np.array(text)[0])
			# a_i,q_values = select_action(config,story,state)
			# print (text)
			# state['curr_summary_ids'].append(int(a_i[0]))
			# state['curr_summary'].append(a_i[1])
			# state['sum_i'] = Sum_i(H_i,state['curr_summary_ids'],q_values)
			# # print (a_i[1])
			# reward_func.get_reward(state['curr_summary'],gold_summ=text)
			# # get_reward(highlights,p)
			
			

def test_logic():
	import torch
	# a = torch.empty(2).uniform_(1, 3)
	a = torch.randint(1,5,size=(2,))
	b = torch.randint(1,5,size=(2,3))
	k = torch.matmul(a,b)
	print (k)
	print (a)
	print (b)

def worker():
	from glob import glob
	files = glob('/home/codexetreme/Desktop/datasets/armageddon_dataset/dm_stories_tokenized/*')
	data_num = len(files)

	examples = []
	for f in files:
		parts = open(f,encoding='latin-1').read().split('\n\n')
		try:
			entities = { line.strip().split(':')[0]:line.strip().split(':')[1].lower() for line in parts[-1].split('\n')}
		except:
			continue
		print (entities)
		sents,labels,summaries = [],[],[]
		# content
		for line in parts[1].strip().split('\n'):
			content, label = line.split('\t\t\t')
			tokens = content.strip().split()
			for i,token in enumerate(tokens):
				if token in entities:
					tokens[i] = entities[token]
			label = '1' if label == '1' else '0'
			sents.append(' '.join(tokens))
			labels.append(label)
		# summary
		for line in parts[2].strip().split('\n'):
			tokens = line.strip().split()
			for i, token in enumerate(tokens):
				if token in entities:
					tokens[i] = entities[token]
			line = ' '.join(tokens).replace('*','')
			summaries.append(line)
		ex = {'doc':'\n'.join(sents),'labels':'\n'.join(labels),'summaries':'\n'.join(summaries)}
		examples.append(ex)
	return examples

if __name__ == '__main__':
	main()
	# test_logic()
	# worker()