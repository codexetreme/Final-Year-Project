import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import create_embedding_matrix
from highway.highway import Highway

def create_emb_layer(weights_matrix, trainable=True):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class SentenceEncoder(nn.Module):
    def __init__(self,target_vocab,vectors,config=None):
        super(SentenceEncoder,self).__init__()
        
        if config is not None:
            self.config = config
        self.embedding_weights_matrix = create_embedding_matrix(self.config,target_vocab,vectors)
        self.embedding_layer,self.num_embeddings,self.embedding_dim = create_emb_layer(self.embedding_weights_matrix)

        print ("self.embedding_layer",self.embedding_layer)
        print ("self.num_embeddings",self.num_embeddings)
        print ("self.embedding_dim",self.embedding_dim)

        Ks = [(1,40),(2,50),(3,50),(4,60),(5,60),(6,70),(7,70)]
        
        self.convs = nn.ModuleList([nn.Conv1d(1,out_channels=out_c,kernel_size=(k,self.config.dataset_options.WORD_DIMENTIONS)) for (k,out_c) in Ks])
    
        self.max_pool = nn.MaxPool1d(kernel_size=self.config.dataset_options.MAX_SENTENCES_PER_DOCUMENT)

        self.highway_layer = nn.ModuleList([Highway(input_size, 3, f=torch.nn.functional.relu) for i,input_size in Ks])

    def conv_and_pool(self, x, conv):
        x = torch.tanh(conv(x)).squeeze(3)  # (N * doc_len, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # (N * doc_len, Co)
        # TODO:
        # We apply the highway network as the paper requires it to be applied if the sentence encoder is a CNN

        return x

    def forward(self,x):
        x = self.embedding_layer(x) # (N,doc_len,sentence_len,dims)
        x = x.unsqueeze(1).view(-1,1,self.config.dataset_options.MAX_SENTENCES_PER_DOCUMENT,self.config.dataset_options.WORD_DIMENTIONS) # (N*doc_len,Ci,sentence_len,dims)

        x = [self.conv_and_pool(x,i) for i in self.convs]
        x = torch.cat(x,dim=1)
        x = x.view(self.config.globals.BATCH_SIZE,-1,400)
        print ('x_shape:', x.shape)
            
        return x



class DocumentEncoder(nn.Module):

    def __init__(self,config=None):
        '''
        D_i => the layer that calulates the document score, it is implemented as presented in the paper.
        '''
        super(DocumentEncoder,self).__init__()
        if config is not None:
            self.config = config
        hidden_size = 50
        self.gru_layer = nn.GRU(input_size=400,hidden_size=hidden_size,num_layers=1,batch_first=True,bidirectional=True)
        self.D_i = nn.Sequential(
                # Due to bi directional, we multipl the hidden size by 2
                nn.Linear(2*hidden_size,2*hidden_size),
                nn.BatchNorm1d(2*hidden_size),
                nn.Tanh()
                )

    def forward(self,x):
        '''
        Output of GRU = output,h_n
        D_i = concat of forward and backward pass of h_n
        H_i = output, these are the vectors that are represent the hidden states
        
        Shapes 
        X = (N,num_sentences, sentence_features[=50]) 50 is the features for the sentence here
        D_i => (N,hidden_size*2[since it is bidirectional])
        H_i => (N,num_sentences,hidden_size*2[since it is bidirectional])
        
        Returns
        D_i,output(ie,H_i)
        '''
        output,h_n = self.gru_layer(x)
        D_i = self.D_i(h_n.view(self.config.BATCH_SIZE,-1))
        print ('Document_encoder_shape',D_i.shape)
        print ('output_shape',output.shape)
        return D_i,output

