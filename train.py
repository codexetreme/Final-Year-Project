import torch.utils.data as datautil
from utils import *
from config import Config
from data_loader import TextDataset
from model import SentenceEncoder
import os


def main():
	config = Config()
	config.paths.GLOVE_PATH = '/home/codexetreme/Desktop/Source/GloVe-1.2/build'
	config.paths.PROCESSED_GLOVE_PATH = '/home/codexetreme/Desktop/datasets/armageddon_dataset'
	config.paths.PATH_TO_CORPUS = '/home/codexetreme/Desktop/datasets/armageddon_dataset/corpus.txt'
	config.paths.PATH_TO_VOCAB_TXT = '/home/codexetreme/Desktop/datasets/armageddon_dataset/vocab.txt'
	config.paths.PATH_TO_DATASET = '/home/codexetreme/Desktop/datasets/armageddon_dataset/cnn_stories_tokenized'
	config.paths.PATH_TO_DATASET_S_VOCAB = '/home/codexetreme/Desktop/datasets/armageddon_dataset/vocab'
	
	# create_vocabulary_from_dataset(config)

	weight_matrix,glove = create_embedding_matrix(config)
	
	# make_target_vocab(config)
	
	word2idx,dataset_vocab = load_target_vocab(config)

	dataset = TextDataset(word2idx,dataset_vocab,config=config)
	data_loader = datautil.DataLoader(dataset=dataset,batch_size=1,num_workers=1,shuffle=False)
	# model = SentenceEncoder(config)


	_i = 1

	for i,(s,h) in enumerate(data_loader):
		# print ("story", s)
		# print ("highlight", h)
		print ("-"*50)
		_i-=1

		if _i<-1:
			break

if __name__ == '__main__':
	main()






