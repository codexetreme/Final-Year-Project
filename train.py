import torch.utils.data as datautil
from utils import *
from config import Configuration
from data_loader import TextDataset
from model import SentenceEncoder,DocumentEncoder
import nonechucks as nc
import os
from tqdm import tqdm
from reward import get_reward

def main():
	configuration = Configuration('project.config')
	config = configuration.get_config_options()
	# print (_.dataset_options)
	# print (c.display())
	# print (_.dataset_options.VOCAB_SIZE)
	# create_vocabulary_from_dataset(config)	
	# make_target_vocab(config)
	
	word2idx,dataset_vectors = load_target_vocab(config)

	dataset = TextDataset(word2idx,dataset_vectors,config=config)
	dataset = nc.SafeDataset(dataset)
	# data_loader = datautil.DataLoader(dataset=dataset,batch_size=64,num_workers=4,shuffle=False)
	# TODO: Set the shuffle to True
	data_loader = nc.SafeDataLoader(dataset=dataset,batch_size=64,num_workers=4,shuffle=True)
	model = SentenceEncoder(target_vocab = word2idx.keys(), vectors = dataset_vectors, config = config)
	doc_enc = DocumentEncoder(config=config)
	# model = DQN(target_vocab = word2idx.keys(),vectors = dataset_vectors,config = config)
	epoch = 0
	for epoch in tqdm(range(epoch,config.globals.NUM_EPOCHS)):
		model.train()
		# doc_enc.train()
		for i,(story,highlights) in tqdm(enumerate(data_loader)):
			p = model(story)
			# get_reward(highlights,p)
			
			

def test_logic():
	import torch
	a = torch.empty(3200,40).uniform_(0, 1)
	b = torch.empty(3200,50).uniform_(0, 1)
	print (torch.cat((a,b),dim=1).shape)


if __name__ == '__main__':
	main()
	# test_logic()
