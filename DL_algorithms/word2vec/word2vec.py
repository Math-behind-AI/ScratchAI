# Implementation of word2vec algorithm from scratch

import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_size, window_size,epochs, learning_rate):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.word2id = {}
        self.id2word = {}

    def generate_training_data(self,text):
        text = text.lower().replace('.', ' .')
        words = text.split(' ')
        for i, word in enumerate(words):
            self.word2id[word] = i
            self.id2word[i] = word
        corpus = [self.id2word[i] for i in range(len(self.id2word))]
        print(corpus)
        training_data=[]
        for i,word in enumerate(corpus):
            sent_len=len(corpus)
            w_target=self.one_hot_encode(corpus[i])
            w_context=[]
            for j in range(i-self.window_size,i+self.window_size+1):
                if j!=1 and j<=sent_len-1 and j>=0:
                    w_context.append(self.one_hot_encode(corpus[j]))
            training_data.append([w_target,w_context])
        return np.array(training_data,dtype=object)

    def one_hot_encode(self,word):
        vector=np.zeros(self.vocab_size)
        vector[self.word2id[word]]=1
        return vector      

    def train(self,training_data):
        self.W1 = np.random.uniform(-1, 1, size=(self.vocab_size, self.embedding_size))
        self.W2 = np.random.uniform(-1, 1, size=(self.embedding_size, self.vocab_size))
        for i in range(self.epochs):
            self.loss=0
            for w_target, w_context in training_data:
                y_pred,h,output_layer=self.forward_pass(w_target)
                EI=np.sum([np.subtract(y_pred,word) for word in w_context],axis=0)
                self.backprop(EI,h,w_target)
                self.loss+=-np.sum([output_layer[np.where(word==1)] for word in w_context])+len(w_context)*np.log(np.sum(np.exp(output_layer)))
            print(f"Epoch:{i},Loss:{self.loss}")     

    def forward_pass(self,one_hot_target):
        h=np.dot(self.W1.T,one_hot_target)
        output_layer=np.dot(self.W2.T,h)
        y_context=self.softmax(output_layer)
        return y_context,h,output_layer            

    def softmax(self,one_hot_target):
        exp=np.exp(one_hot_target)
        return exp/np.sum(exp)

    def backprop(self,EI,h,w_target):
        dL_dW2=np.outer(h,EI)
        dL_dW1=np.outer(w_target,np.dot(self.W2,EI.T))
        self.W1=self.W1-(self.learning_rate*dL_dW1)
        self.W2=self.W2-(self.learning_rate*dL_dW2) 

    def word_vec(self,word):
        w_index=self.word2id[word]
        v_word=self.W1[w_index]
        return v_word       

    def vec_similarity(self,word,top_n):
        v_word1=self.word_vec(word)
        word_sim={}
        for i in range(self.vocab_size):
            v_target=self.W1[i]
            theta_sum=np.dot(v_word1,v_target)
            theta_den=np.linalg.norm(v_word1)*np.linalg.norm(v_target)
            theta=theta_sum/theta_den
            word=self.id2word[i]
            word_sim[word]=theta
        words_sorted=sorted(word_sim.items(),key=lambda x:x[1],reverse=True)
        for word,similarity in words_sorted[:top_n]:
            print(word,similarity)


res=Word2Vec(17,2,2,10,0.01)
text="Recurrent neural network based language model has been proposed to overcome certain limitations of the feedforward NNLM"
training_data=res.generate_training_data(text)
print(training_data)
res.train(training_data)
res.vec_similarity('limitations',3)