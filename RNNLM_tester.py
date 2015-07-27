from IndexedCorpus import *
filename = '2.txt'
Laura_Corpus_Indexed = IndexedCorpus(filename, SeenTreshold = 5)

import cPickle
f = file(filename[0]+'.txt.objects.save', 'rb')
W,V,U,s0 = cPickle.load(f)
f.close()
################################
import warnings
warnings.filterwarnings("ignore")
################################
import sys
import theano
import numpy
from scipy import sparse
from theano import tensor as T
import time

s = {
    'vocSize': Laura_Corpus_Indexed.vocSize, # include <UNK>
}

trainSetConcatenatedArray = array(list(itertools.chain.from_iterable(Laura_Corpus_Indexed.train_set)))
trainX,trainY = Idx2DataSet(trainSetConcatenatedArray)
testSetConcatenatedArray = array(list(itertools.chain.from_iterable(Laura_Corpus_Indexed.test_set)))
testX,testY = Idx2DataSet(testSetConcatenatedArray)

n_val = s['vocSize']
# v_val = np.asarray([1,0,3])
w_idxs = T.ivector()
z = theano.tensor.zeros((w_idxs.shape[0], n_val))
one_hot = theano.tensor.set_subtensor(z[theano.tensor.arange(w_idxs.shape[0]), w_idxs], 1)
emb = theano.function([w_idxs], one_hot)
# print emb(v_val)

# w_idxs = T.ivector()
w_t = T.fmatrix() # it is a sequence of vectors
y_t = T.fmatrix() # y0 is just a vector since scan has only to provide y[-1]

def oneStep(w_t, s_tm1,W,V,U):
            s_t = T.nnet.sigmoid(T.dot(w_t, U) + T.dot(s_tm1, W))
            y_t = T.nnet.softmax(T.dot(s_t, V))
            return [s_t, y_t]

[s_vals, y_vals], _ = theano.scan(fn=oneStep,
                          sequences = dict(input=w_t, taps=[0]),
                          outputs_info = [s0, None], # corresponds to return type of fn
                          non_sequences = [W,V,U] )

p_y_given_x = y_vals[:,0,:] # (BatchSize, 1L, vocSize) //  (7L, 1L, 56L)
#y_pred = T.argmax(p_y_given_x,axis=1)
# cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x,emb[y_idxs]))
# gradients = T.grad(cost=cost, wrt=params)
# updates = OrderedDict(( p, p-s['learingRate']*g ) for p, g in zip(params , gradients))
# train = theano.function(inputs=[w_idxs,y_idxs], outputs=cost, updates=updates, allow_input_downcast=True)
#predict = theano.function(inputs=[w_idxs], outputs=y_pred, allow_input_downcast=True)
prob_y_given_x = theano.function(inputs=[w_t], outputs=p_y_given_x, allow_input_downcast=True)
################################
print perplexity(testX,testY,prob_y_given_x,emb)