import os
import bcolz
import numpy as np
import pickle
import subprocess
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	def pad_char(a):
		 a = a.strip() + ' .\n'
		 return a
	highlights = [ pad_char(h) for h in highlights if len(h) > 0]
	return story, highlights



def process_glove(glove_path,dims):
	'''
	This method writes 2 pkl files that store the word list from the glove 
	vectors (~400K words) and a corresponding file that stores each words index
	
	:param glove_path: path to glove file
	:param dims: the dimentions to use

	this makes the total path as {glove_path}/glove.6B.{dims}d.txt

	NOTE: bcolz stores a .dat folder that is used for compressing the large number of 
	words that are present in the glove file

	Run this function once to generate the respective pickle files.

	'''
	path = os.path.join(glove_path,'glove.6B.{dims}d.txt'.format(dims=dims))
	dat_file_name = os.path.join(glove_path,'glove.6B.{dims}d.dat'.format(dims=dims))
	print (path)
	words = []
	idx = 0
	word2idx = {}
	vectors = bcolz.carray(np.zeros(1), rootdir=dat_file_name, mode='w')

	with open(path, 'rb') as f:
		for l in f:
			line = l.decode().split()
			word = line[0]
			words.append(word)
			word2idx[word] = idx
			idx += 1
			vect = np.array(line[1:]).astype(np.float)
			vectors.append(vect)
		
	vectors = bcolz.carray(vectors[1:].reshape((400000, dims)), rootdir=dat_file_name, mode='w')
	vectors.flush()
	pickle.dump(words, open(f'{glove_path}/6B.100_words.pkl', 'wb'))
	pickle.dump(word2idx, open(f'{glove_path}/6B.100_idx.pkl', 'wb'))

# process_glove('/home/codexetreme/Desktop/datasets/',100)


def load_glove_vectors(config):
	glove_path = config.paths.PROCESSED_GLOVE_PATH
	vectors = bcolz.open(os.path.join(glove_path,'glove.6B.100d.dat'))[:]
	words = pickle.load(open(os.path.join(glove_path,'6B.100_words.pkl'), 'rb'))
	word2idx = pickle.load(open(os.path.join(glove_path,'6B.100_idx.pkl'), 'rb'))
	glove = {w: vectors[word2idx[w]] for w in words}
	return glove
# g = test_glove_vectors('/home/codexetreme/Desktop/datasets/armageddon_dataset/')

# print(g['shanky'])

def make_target_vocab(config):
	'''
	This function reads the vocab.txt that was generated by the vocab_count program in glove.
	
	Simple cleaning mechanisms are employed:
		> converting the words to lowercase
		> TODO: (functions like removing stop words can be added)
	
	The words are then written to disk with their corresponding glove vectors and 
	a word2idx file is also written, this file contains the integer mapping of the words to a specific index
	'''

	if not os.path.exists(config.paths.PATH_TO_DATASET_S_VOCAB):
		os.makedirs(config.paths.PATH_TO_DATASET_S_VOCAB)
	

	dat_file_name = os.path.join(config.paths.PATH_TO_DATASET_S_VOCAB,'target_vocab.dat')
	vectors = bcolz.carray(np.zeros(1), rootdir=dat_file_name, mode='w')
	
	glove = load_glove_vectors(config)

	targetwords2id = {}
	targetwords2id['<PAD>'] = 0
	targetwords2id['<START>'] = 1
	targetwords2id['<END>'] = 2
	targetwords2id['<UNK>'] = 3
	
	# for the vectors for these 4 words above, since these arent present in the glove vectors
	vect = np.random.normal(scale=0.6,size=(4*config.WORD_DIMENTIONS, ))
	vectors.append(vect )

	idx = 4
	with open(config.paths.PATH_TO_VOCAB_TXT) as file:
		for line in file.readlines():
			word = line.split(' ')[0]
			targetwords2id[word] = idx
			idx += 1
			try:
				vect = np.array(glove[word]).astype(np.float)
			except KeyError:
				vect = np.random.normal(scale=0.6, size=(config.WORD_DIMENTIONS, ))
			finally:
				vectors.append(vect)
	vectors = bcolz.carray(vectors[1:].reshape((-1, config.WORD_DIMENTIONS)), rootdir=dat_file_name, mode='w')
	vectors.flush()
	pickle.dump(list(targetwords2id.keys()), open(os.path.join(config.paths.PATH_TO_DATASET_S_VOCAB,'words.pkl'), 'wb'))
	pickle.dump(targetwords2id, open(os.path.join(config.paths.PATH_TO_DATASET_S_VOCAB,'words2idx.pkl'), 'wb'))


def load_target_vocab(config):
	path = config.paths.PATH_TO_DATASET_S_VOCAB
	vectors = bcolz.open(os.path.join(path,'target_vocab.dat'))[:]
	words = pickle.load(open(os.path.join(path,'words.pkl'), 'rb'))
	word2idx = pickle.load(open(os.path.join(path,'words2idx.pkl'), 'rb'))
	dataset_vectors = {w: vectors[word2idx[w]] for w in words}
	return word2idx,dataset_vectors

def create_vocabulary_from_dataset(config):
	cmd = [os.path.join(config.paths.GLOVE_PATH,"vocab_count"), "-min-count",str(config.MIN_VOCAB_COUNT)
		,"-max-vocab",str(config.VOCAB_SIZE)]
	corpus_file = open(config.paths.PATH_TO_CORPUS,'r')
	vocab_file = open(config.paths.PATH_TO_VOCAB_TXT,'w+')
	try:
		subprocess.check_call(cmd,stdin=corpus_file,stdout=vocab_file)
	finally:
		corpus_file.close()
		vocab_file.close()


def create_embedding_matrix(config,target_vocab,vectors):
	matrix_len = len(target_vocab)
	weights_matrix = np.zeros((matrix_len, config.dataset_options.WORD_DIMENTIONS))
	words_found = 0
	for i, word in enumerate(target_vocab):
		try: 
			weights_matrix[i] = vectors[word]
			words_found += 1
		except KeyError:
			weights_matrix[i] = np.random.normal(scale=0.6, size=(config.dataset_options.WORD_DIMENTIONS, ))
	return weights_matrix



'''
These 3 functions are used to create a nested version of SimpleNamespace when 
we need to Namespace nested dictionaries

Example:

    >>> mydict = {'a':123, 'b':{'c':234,'d':345}}
    >>> ns = wrap_namespace (mydict)
    >>> ns.b.c
    234
    >>> ns.a
    123

'''
from functools import singledispatch
from types import SimpleNamespace
@singledispatch
def wrap_namespace(ob):
    return ob

@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{k: wrap_namespace(v) for k, v in ob.items()})

@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(v) for v in ob]