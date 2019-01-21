import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
import itertools
from utils import make_target_vocab,load_glove_vectors,split_story
class TextDataset(): 
	''' Dataset Loader for a text dataset'''
		
	def get_list_of_docs(self,root_dir):
		return os.listdir(root_dir)

	def __init__(self,word2idx,dataset_vocab, config=None,run_type=None, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		# self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = os.path.abspath(config.paths.PATH_TO_DATASET)
		self.transform = transform
		self.word2idx = word2idx
		self.dataset_vocab = dataset_vocab
		self.file_list = self.get_list_of_docs(self.root_dir)
		if config is not None:
			self.config = config
		
	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		_path = os.path.join(self.root_dir,self.file_list[idx])
		
		with open(_path) as file:
			text = file.read()
			text = text.replace('-LRB-','(')
			text = text.replace('-RRB-',')')
			story, highlights = split_story(text)
		# Get all the 
		s = np.array(story.splitlines())
		# Pick all non empty sentences
		k = s[s!='']
		# Split all the sentences on white space, to get the words
		k = np.char.split(k)
		# Pad the resulting array of lists. Each list represents the words of that sentence
		# This function, pads all the lists to the length of the max list in the array
		k = np.array(list(itertools.zip_longest(*k, fillvalue=self.config.PADDING_SEQ))).T
		# In case, there are sentences with less than words, we pad all the lists with 55 pad tokens
		k = np.pad(k,((0,0),(0,55)),'constant',constant_values=self.config.PADDING_SEQ)
		# print (k[0:2])
		# Here we first strip the documents to required length
		k = k[:self.config.MAX_SENTENCES_PER_DOCUMENT]
		# Here the sentences are now stripped to required length, thus resulting in the final shape of (num_sentences,50)
		k = k[:,:self.config.MAX_WORDS_PER_SENTENCE]
		print (k.shape)
		print (k)
		# print ("post cut",k.shape)
		return story, highlights

