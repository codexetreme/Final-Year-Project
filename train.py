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
			print ('EPSILON Decayed to : ',config.dqn.EPS_START)

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
			next_state = state
			prev_r_i,r_i = 0
			# locking to 10 for simplicity purposes
			for i in count(config.dqn.SUMMARY_LENGTH):
				iter = iter + 1

				# if i>20 : break

				story['tensor'] = story['tensor'].to(device)
				highlights['tensor'] = highlights['tensor'].to(device)
				# sentence representation are calculated as no grad because we dont want to disturb / update the weights 
				with torch.no_grad():
					H_i,D_i,x = policy_net(story['tensor'][:,:len(story['raw'])-1])

				a_i,q_values = select_action(config,story,state)
				
				next_state['curr_summary_ids'].append(int(a_i[0]))
				next_state['curr_summary'].append(a_i[1])
				next_state['sum_i'] = Sum_i(H_i,state['curr_summary_ids'],q_values)
				r_i = reward_func.get_reward([next_state['curr_summary']],gold_summ=[[highlights['raw']]],**{'prev_score':prev_r_i,'config':config})
				prev_r_i = r_i
				# checks if we are close to the summ length part 
				done = check_done(config,next_state)
				if done:
					next_state = None
				# TODO: check which a_i has to be loaded , a_i[0] or a_i[1] or just a_i
				memory.push(state, H_i[a_i[0]], next_state, r_i)
				state = next_state 
				optimize_model(config)


				if done:
					break


def check_done(config,summary):
	l = len(str(next_state['curr_summary']))			
	if l > config.globals.SUMMARY_LENGTH:
		return False
	return True
	

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
def optimize_model(config):
    if len(memory) < 64: # Batch size
        return
    transitions = memory.sample(config.globals.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(config.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * config.dqn.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

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