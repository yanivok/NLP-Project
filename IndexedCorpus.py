import numpy as np
from scipy.sparse import csc_matrix
import itertools
from collections import OrderedDict
from numpy import array
from nltk import FreqDist
import cPickle
from math import log
################################### Indexed Corpus Class ######################################


class IndexedCorpus(object):

    def __init__(self, filename, SeenTreshold = 1):
        self.corpus = []
        file = open(filename, 'r')

        num_lines = sum(1 for line in file)
        file.close()

        file = open(filename, 'r')
        count_lines = 0
        for line in file:
            new_line = line.split('\n')[0]
            words = new_line.split(' ')
            words.insert(0,'<S>')
            self.corpus.append(words)
            count_lines += 1
            if count_lines >= int(0.8*num_lines):
                break

        chain_of_words = itertools.chain.from_iterable(self.corpus)
        self.words_corpus = list(chain_of_words)  # concatenate the lists


        self.fdist = FreqDist(self.words_corpus)

        self.WordsSeenBelowTreshold = []
        self.Vocabulary = []

        for w in self.fdist.keys():
            if self.fdist[w] <= SeenTreshold:
                self.WordsSeenBelowTreshold.append(w)
            else:
                self.Vocabulary.append(w)

        self.words_corpus = ['<UNK>' if w in self.WordsSeenBelowTreshold else w for w in self.words_corpus]
        self.idx2word = dict(enumerate(OrderedDict.fromkeys(self.words_corpus)))
        self.words2idx = dict(zip(self.idx2word.values(), self.idx2word.keys()))
        self.vocSize = len(self.words2idx)

        # add to corpus the 30% rest of lines:
        for line in file:
            new_line = line.split('\n')[0]
            words = new_line.split(' ')
            words.insert(0,'<S>')
            self.corpus.append(words)
        file.close()

        self.only_numbers_corpus = []
        for sentence in self.corpus:
            indexed_sentence = map(lambda x: self.word2index(x), sentence)
            self.only_numbers_corpus.append(array(indexed_sentence))

        self.train_set = self.only_numbers_corpus[:int(len(self.only_numbers_corpus)*0.8)]
        # self.valid_set = self.only_numbers_corpus[int(len(self.only_numbers_corpus)*0.7):int(len(self.only_numbers_corpus)*0.8)]
        self.test_set  = self.only_numbers_corpus[int(len(self.only_numbers_corpus)*0.8):]

        # self.OneHotRepCorpus = []
        # for sentence in self.only_numbers_corpus:
        #     OneHot_sentence = map(lambda x: self.OneHotRep(x), sentence)
        #     self.OneHotRepCorpus.append(OneHot_sentence)

    def index2word(self, index):
        try:
            return self.idx2word[index]
        except:
            print "not a valid index! please insert index between 0 and " + str(self.vocSize-1)


    def word2index(self, word):
        if word in self.Vocabulary:
            return self.words2idx[word]
        else:
            return self.words2idx['<UNK>']

    def OneHotRep(self,index):
        vector = csc_matrix((1, self.vocSize), dtype=np.int8).toarray()
        vector[0,index] = 1
        return vector
################################ Assistant Methods #######################################################

def ParamSave(parameters,filename):
    f = file(filename + '.objects.save', 'wb')
    cPickle.dump(parameters, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def perplexity(testX,testY,prob_y_given_x,emb):
    matrix = prob_y_given_x(emb(testX))
    probs = [matrix[x] for x in zip(range(0,len(testY)),testY)]

    sum_minus_logprobs = -sum([log(x,10) for x in probs]) # Sigma [ log2( P(word | model) ) ]
    total_words = len(testY)
    # total_words_and_start_simbols = total_words + sum(testY == 0) # TODO: check if 0 is the <s> symbol
    cross_entropy = sum_minus_logprobs / float(total_words)
    PPL = 10**cross_entropy # PPL = 2^Entropy
    return PPL

def Idx2DataSet(ConcatenatedArray):
    trainX,trainY = ConcatenatedArray[:-1], ConcatenatedArray[1:]
    X = [(x,y) for (x,y) in zip(trainX,trainY) if y!=0]
    trainX = []; trainY = []
    for instance in X:
        trainX.append(instance[0])
        trainY.append(instance[1])
    trainX = array(trainX); trainY = array(trainY)
    return trainX, trainY
# def window(sentence):
#     padded_sentence = [-1] + sentence
#     if len(padded_sentence) % 2 == 0:
#         out = [padded_sentence[i:(i + 2)] for i in range(len(sentence))]
#     else:
#         padded_sentence = [-1] + sentence + [-1]
#         out = [padded_sentence[i:(i + 2)] for i in range(len(sentence))]
#     return out
#
# def flatten(l):
#     temp =itertools.chain.from_iterable(l)
#     return list(temp)
#################################### Defining Hierarchies ###############################################

# Laura_Corpus_Indexed = IndexedCorpus('1.txt')
# CNN_World_Corpus_indexed = IndexedCorpus('2.txt')
# Combined_CNN_Corpus_indexed = IndexedCorpus('3.txt')
#
#for example:
# print Laura_Corpus_Indexed.only_numbers_corpus[100]
# print Laura_Corpus_Indexed.idx2word
# #and the opposite dictionary:
# print Laura_Corpus_Indexed.words2idx
# #only numbers laura corpus
# print Laura_Corpus_Indexed.only_numbers_corpus
# #out [input,output]
# print window(flatten(Laura_Corpus_Indexed.only_numbers_corpus))
# #if you want One-Hot representation so:
# print window(flatten(Laura_Corpus_Indexed.OneHotRepCorpus))

#[[-1, array([ 0.,  0.,  1., ...,  0.,  0.,  0.])], [array([ 0.,  0.,  1., ...,  0.,  0.,  0.]), array([ 0.,  0.,  0.,...
## we need to deal with the case of window padding with -1 on start
