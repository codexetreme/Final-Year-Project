import torch
import torch.utils.data as datautil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np

from utils import split_story, make_target_vocab, load_glove_vectors
from config import Config

import os
import sys
import subprocess
class TextDataset(): 
	''' Dataset Loader for a text dataset'''
		
	def create_embedding_matrix(self,emb_dim):
		matrix_len = self.config.VOCAB_SIZE
		weights_matrix = np.zeros((matrix_len, emb_dim))
		words_found = 0
		target_vocab = make_target_vocab(self.config)
		glove = load_glove_vectors(self.config)
		for i, word in enumerate(target_vocab):
		    try: 
		        weights_matrix[i] = glove[word]
		        words_found += 1
		    except KeyError:
		        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

	def get_list_of_docs(self,root_dir):
		return os.listdir(root_dir)

	def __init__(self, config=None,run_type=None, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		# self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = os.path.abspath(config.paths.PATH_TO_DATASET)
		self.transform = transform
		self.file_list = self.get_list_of_docs(self.root_dir)
		print ("total length ",len(self.file_list))
		print ("first 5 file names",self.file_list[:5])
		if config is not None:
			self.config = config
		
	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		file = open(os.path.join(self.root_dir,self.file_list[idx]), encoding='utf-8')
		text = file.read()
		file.close()
		story, highlights = split_story(text)
		return story, highlights


config = Config()
config.paths.GLOVE_PATH = '/home/codexetreme/Desktop/Source/GloVe-1.2/build'
config.paths.PROCESSED_GLOVE_PATH = '/home/codexetreme/Desktop/datasets/armageddon_dataset'
config.paths.PATH_TO_CORPUS = '/home/codexetreme/Desktop/datasets/armageddon_dataset/corpus.txt'
config.paths.PATH_TO_VOCAB_TXT = '/home/codexetreme/Desktop/datasets/armageddon_dataset/vocab.txt'
config.paths.PATH_TO_DATASET = '/home/codexetreme/Desktop/datasets/armageddon_dataset/cnn_stories_tokenized'
dataset = TextDataset(config=config)
data_loader = datautil.DataLoader(dataset=dataset,batch_size=1,num_workers=4,shuffle=False)

create_vocabulary_from_dataset()
dataset.create_embedding_matrix(100)


# _i = 1

# for i,(s,h) in enumerate(data_loader):
# 	print ("i = ", i)
# 	print ("story", s)
# 	print ("highlight", h)
# 	print ("-"*50)
# 	_i-=1
# 	if _i<-1:
# 		break