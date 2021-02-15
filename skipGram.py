from __future__ import division
import argparse
import pandas as pd
 
# useful stuff
import pickle
from collections import defaultdict,Counter
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
 
 
__authors__ = ['Parth MITTAL','Ignacio MORENO','Sarthak RAISURANA','Elias SELMAN']
__emails__  = ['parth.mittal@student-cs.fr','ignacio.moreno@student-cs.fr','sarthak.raisurana@student-cs.fr','elias.selman@student-cs.fr']

def text2sentences(path):
    '''
    Tokenizer to remove the special characters,white-spaces and empty words
    '''
    sentences = []
    numb = [i for i in range(10) ]
    spcl_char = ['|','?','_','!','¶','€',':',']','\t','^',
                 '/','*','}','$','~','&','{','[','(','"',
                 '`','=',';','@','+','%',')','>','.',',',
                 '\\','"','£','\n','-']
    with open(path, encoding='utf8') as f:
        for l in f:
            #remove numbers
            for n in numb:
                l = l.replace(str(n),'')
                
            #remove special characters (except ')
            for chr in spcl_char:
                l=l.replace(chr,'')
                
            #split
            l= l.lower().split()
            #remove empty words 
            l = [x.strip() for x in l if len(x.strip())>1]
            
            #Appending 
            sentences.append(l)
            
    return sentences
 
 
def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs
 
  
 
class SkipGram:
    '''
    SkipGram Model with Negative sampling
    Hyperparameters: The hyperparameters in our Skip-gram model are listed below:
        minCount(default:1): minimum frequency for words to be considered in the vocabulary set.
        negativeRate(default:5): number of negative sample words for each pair of target-context using sampling().
        winSize(default:5): Context window size. 
        nEmbed(default:300): Size of the embedding(num of features to define word in embedding).
        Epochs(default:10): Number of iterations of the model while training.
        Learning rate(default:0.2): step size for weight updates.
        alpha(default:0.75): for unigram distribution
        
        Embeddings: There are two embeddings used for input and output, referred as word (w;size |V| x N) 
        and context (c;size N x |V| ) embedding respectively.
        
    The class has save() and load() for saving and loading the model
    '''
    def __init__(self, sentences, nEmbed= 300, negativeRate=5,
                 winSize = 5, minCount = 1,epochs = 10, alpha =0.75, lr =0.2):
        
        #Initializing parameters
        self.negativeRate = negativeRate
        self.nEmbed = nEmbed
        self.winSize = winSize
        self.minCount = minCount
        self.epochs = epochs
        self.lr = lr
        self.alpha = 0.75
        
        
        #Building training Corpus
        self.trainset = sentences # set of sentences
        self.concat_sentence = np.concatenate(sentences)
        self.all_unique_words,self.frequency = np.unique(self.concat_sentence, return_counts= True)
        self.all_unique_words_frequency = zip(self.all_unique_words,self.frequency)
        
        #removing words
        self.vocab,self.freq_f = list(zip(*list(filter(lambda x: x[1] >= self.minCount, self.all_unique_words_frequency))))
        
        #setting up w2id/id2w  mapping
        self.w2id = {word:i for i, word in enumerate(self.vocab)} # word to ID mapping
        self.id2w = {i:word for word,i in self.w2id.items()}
        
        # Calculating the probs for unigram distribution (alpha)
        self.freq_f = np.array(self.freq_f)
        self.unigram_prob = self.freq_f ** self.alpha/ np.sum(self.freq_f ** self.alpha)
        
        # Initializing Embeddings
        self.w = np.random.randn(len(self.vocab), self.nEmbed) *0.001 #np.random.randn(len(self.vocab), ) # |V| x N 
        self.c = np.random.randn(len(self.vocab), self.nEmbed) *0.001 #np.random.randn(self.nEmbed, len(self.vocab) ) #  N x |V|
        
        # loss and support variables
        self.loss = []
        self.accLoss = 0
        self.trainWords = 0
  
    
    def sample(self, omit):
        """
        samples negative words, ommitting those in set omit
        """
        #getting list of indices to be ommited
        self.omit= list(omit)
        random_sample_id=[]
        #Sampling negativeRate num of negative samples
        while len(random_sample_id)<self.negativeRate:
            #sampling from Unigram distribution
            pick = np.random.choice(list(self.w2id.values()),size=1, replace=False,p= self.unigram_prob)
            if pick[0] not in self.omit:
                #removing indices to be omitted
                random_sample_id.append(pick[0])
        
        return random_sample_id
 
    def train(self):
        '''
        Training function running through each filtered sentences in training corpus
        and calls TrainWord function for each word with one of its context words and negative samples
        in the WinSize
        '''
        for epoch in range(self.epochs):
          

          for counter, sentence in enumerate(self.trainset):
              sentence = list(filter(lambda word: word in self.vocab, sentence))
          
              
              for wpos, word in enumerate(sentence):
                  wIdx = self.w2id[word]
                  winsize = np.random.randint(self.winSize) + 1
                  start = max(0, wpos - winsize)
                  end = min(wpos + winsize + 1, len(sentence))
                  
  
                  for context_word in sentence[start:end]:
                      ctxtId = self.w2id[context_word]
                      if ctxtId == wIdx: continue
                      negativeIds = self.sample({wIdx, ctxtId})
                      self.trainWord(wIdx, ctxtId, negativeIds)
                      self.trainWords += 1
              if counter % 200 == 0:
                print(' > training {} of {}'.format(counter, len(self.trainset)))

          
          print(' > training Epoch {}: Loss = {}'.format(epoch+1,self.accLoss))

          self.loss.append(self.accLoss)
          #self.save('model_full_{}'.format(epoch))
          self.accLoss = 0
          
 
    def trainWord(self, wordId, contextId, negativeIds):
        '''
        Calculating gradietns and Updating each word embedding
        '''
        

        #Sigmoid function
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        
        #Indexing the embedding     
        h = self.w[wordId]
        c_pos = self.c[contextId]
        c_neg = self.c[negativeIds]

        #out
        out_pos = sigmoid(c_pos.dot(h))
 
            
        # loss
        totalLoss=  -(np.log(out_pos) + np.sum( np.log(sigmoid(- c_neg.dot(h))) ))
    
 
        # Backward propogation
        # Calculating Gradients
            
        # context word
        # gradient for word embedding 
        grad_w = (out_pos-1) * c_pos
            
        # updating gradient for context embedding
        grad_c_pos = (out_pos-1) * h
        self.c[contextId] -= self.lr *  grad_c_pos
 
        # gradient for context embedding for negative word          
        for i,c in enumerate(c_neg):
          out_c_neg1 = sigmoid(c.dot(h))
          # calculating and adding gradient for word embedding for negative word
          grad_w += out_c_neg1 * c
              
          # updating gradient for context embedding for negative word
          grad_c_neg = out_c_neg1 * h
          self.c[negativeIds[i]] -= self.lr * grad_c_neg
            
            
        # updating word embedding for input word
        self.w[wordId] = self.w[wordId] - self.lr * grad_w
            
        # Adding the word loss to total loss
        self.accLoss += totalLoss
      
    def save(self, path):
            """
            save the data to file
            """
            data = {'w2id': self.w2id,
                    'w': self.w,
                    'c': self.c,
                    'negativeRate': self.negativeRate,
                    'nEmbed': self.nEmbed,
                    'winSize': self.winSize,
                    'minCount': self.minCount,
                    'accLoss': self.accLoss,
                    'loss':self.loss}

            with open(path, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
 
 
    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        if word1 in list(self.w2id.keys()):
          w1_emb = self.w[self.w2id[word1]] + self.c[self.w2id[word1]]
        else:
          return 0.5
        
        if word2 in list(self.w2id.keys()):
          w2_emb = self.w[self.w2id[word2]] + self.c[self.w2id[word2]]
        else:
          return 0.5
        
 
        cosine = np.sum(w1_emb*w2_emb) / (np.linalg.norm(w1_emb)*np.linalg.norm(w2_emb))
        return cosine
    
    @staticmethod
    def load(path):
        """
        Load the parameters for testing
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        sg = SkipGram(sentences=[['list','maker']*100],
                      nEmbed=data['nEmbed'],
                      negativeRate=data['negativeRate'],
                      winSize=data['winSize'],
                      minCount=data['minCount'])
        sg.w = data['w']
        sg.c = data['c']
        sg.w2id = data['w2id']
        sg.loss = data['loss']
        sg.accLoss = data['accLoss']
        return sg


    def similar_words(self,word, numwords =10):
        '''
        Get the most similar words to a word given as input
        '''
        similiar_words_dict = {}
        for w in self.w2id.keys():
            if(w != word):
                similiar_words_dict[w] = self.similarity(word, w)
        similiar_words_dict = sorted(similiar_words_dict.items(), key=lambda x: x[1], reverse = True)[:numwords]
        
        wl=[]
        siml = []
        for (i,v) in similiar_words_dict:
            wl.append(i)
            siml.append(v)
         
        result_df = pd.DataFrame({'word':wl,'simil':siml})
        
        return result_df 
 

        
        
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')

	opts = parser.parse_args()

	if not opts.test:
		sentences = text2sentences(opts.text)
		sg = SkipGram(sentences, minCount= 10, lr=0.1, epochs=1)
		sg.train()
		sg.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		sg = SkipGram.load(opts.model)
		for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(sg.similarity(a,b))
            
           