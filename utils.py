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
	highlights = [h.strip() for h in highlights if len(h) > 0]
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
	# output_file = open('test.txt','w+')
	# subprocess.call(['awk','{print $1}',config.paths.PATH_TO_VOCAB_TXT],stdout=output_file)
	# subprocess.call(['mv','test.txt',config.paths.PATH_TO_VOCAB_TXT])
	words = []
	with open(config.paths.PATH_TO_VOCAB_TXT) as file:
		for line in file.readlines():
			words.append(line.split(' ')[0])
	return words
	# output_file.close()	

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