import torch
import torch.utils.data as datautil
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import split_story

class TextDataset(): 
	''' Dataset Loader for a text dataset'''
	
	

	def get_list_of_docs(self,root_dir):
		return os.listdir(root_dir)

	def __init__(self, root_dir,run_type=None, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		# self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = os.path.abspath(root_dir)
		self.transform = transform
		self.file_list = self.get_list_of_docs(self.root_dir)
		print ("total length ",len(self.file_list))
		print ("first 5 file names",self.file_list[:5])

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		file = open(os.path.join(self.root_dir,self.file_list[idx]), encoding='utf-8')
		text = file.read()
		file.close()
		story, highlights = split_story(text)
		return story, highlights

dataset = TextDataset("/home/codexetreme/Desktop/datasets/cnn_stories_tokenized")
data_loader = datautil.DataLoader(dataset=dataset,batch_size=1,num_workers=4,shuffle=False)

_i = 1

for i,(s,h) in enumerate(data_loader):
	print ("i = ", i)
	print ("story", s)
	print ("highlight", h)
	print ("-"*50)
	_i-=1
	if _i<-1:
		break