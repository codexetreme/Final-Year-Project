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

	def get_word2id(self,word):
		try:
			id = self.word2idx[word]
		except KeyError:
			id = self.word2idx['<UNK>']
		
		return id


	def make_tensor_from_string(self,context):
		# Separate the lines in the story 
		context = np.array(context.splitlines())
		# Pick all non empty sentences and
		# Split all the sentences on white space, to get the words
		context = np.char.split(context[context!=''])
		# Pad the resulting array of lists. Each list represents the words of that sentence
		# This function, pads all the lists to the length of the max list in the array
		context = np.array(list(itertools.zip_longest(*context, fillvalue=self.config.PADDING_SEQ))).T
		# In case, there are sentences with less than words, we pad all the lists with 55 pad tokens
		context = np.pad(context,((0,55),(0,55)),'constant',constant_values=self.config.PADDING_SEQ)
		# Here the document is stripped to required length,  thus resulting in the final shape of (num_sentences,50)
		context = context[:self.config.MAX_SENTENCES_PER_DOCUMENT,:self.config.MAX_WORDS_PER_SENTENCE]

		context_idxs = torch.tensor([self.get_word2id(w) for w in context.reshape(-1)],dtype=torch.long) 
		# for i,word in enumerate(context):
		return context_idxs.view(-1,self.config.MAX_WORDS_PER_SENTENCE)

	def __init__(self,word2idx,dataset_vocab, config=None,run_type=None, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
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
			text = text.lower()
			story, highlights = split_story(text)
			highlights = ''.join(highlights)

		story_tokens = self.make_tensor_from_string(story)
		highlights_tokens = self.make_tensor_from_string(highlights)
			 
		return story_tokens,highlights_tokens

