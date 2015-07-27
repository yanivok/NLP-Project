import warnings
warnings.filterwarnings("ignore")

import sys
import theano
import numpy
from scipy import sparse
from theano import tensor as T
from IndexedCorpus import *
import time
# from theano import config
# config.exception_verbosity = 'high'
filename = '1.txt'
Laura_Corpus_Indexed = IndexedCorpus(filename, SeenTreshold = 5)

s = {
    'nEpochs': 50,
    'miniBatch': 256,
    'learingRate': 2,
    'verbose': True,
    'decay': True,
    'vocSize': Laura_Corpus_Indexed.vocSize, # include <UNK>
    'hiddenLayes': 30,#100
    'BestEpoch': 0
}

trainSetConcatenatedArray = array(list(itertools.chain.from_iterable(Laura_Corpus_Indexed.train_set)))
trainX,trainY = Idx2DataSet(trainSetConcatenatedArray)
testSetConcatenatedArray = array(list(itertools.chain.from_iterable(Laura_Corpus_Indexed.test_set)))
testX,testY = Idx2DataSet(testSetConcatenatedArray)

W = theano.shared(numpy.random.uniform(-1.0, 1.0, (s['hiddenLayes'],s['hiddenLayes'])).astype(theano.config.floatX))
V = theano.shared(numpy.random.uniform(-1.0, 1.0, (s['hiddenLayes'],s['vocSize'])).astype(theano.config.floatX))
U = theano.shared(numpy.random.uniform(-1.0, 1.0, (s['vocSize'],s['hiddenLayes'])).astype(theano.config.floatX))
s0= theano.shared(numpy.random.uniform(-1.0, 1.0, (s['hiddenLayes'])).astype(theano.config.floatX))
# emb = theano.shared(sparse.eye(s['vocSize'],dtype=numpy.int8).toarray().astype(theano.config.floatX))
params = [W,V,U,s0]

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
cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x,y_t))
gradients = T.grad(cost=cost, wrt=params)
updates = OrderedDict(( p, p-s['learingRate']*g ) for p, g in zip(params , gradients))
train = theano.function(inputs=[w_t,y_t], outputs=cost, updates=updates, allow_input_downcast=True)
#predict = theano.function(inputs=[w_idxs], outputs=y_pred, allow_input_downcast=True)
prob_y_given_x = theano.function(inputs=[w_t], outputs=p_y_given_x, allow_input_downcast=True)

######################################################################
best_TestPPL = numpy.inf
for epoch in xrange(s['nEpochs']):
    s['CurrentEpoch'] = epoch
    tic = time.time()

    for start,end in zip(range(0,len(trainX),s['miniBatch']),range(s['miniBatch'],len(trainX),s['miniBatch'])):
        cost = train(emb(trainX[start:end]), emb(trainY[start:end]))

        if s['verbose']:
            print '[learning] epoch %i >> %2.2f%%'%(epoch,(end+1)*100./len(trainX)),'completed in %.2f (sec) <<\r'%(time.time()-tic),
            sys.stdout.flush()
    currentTestPPL = perplexity(testX,testY,prob_y_given_x,emb)

    if currentTestPPL < best_TestPPL:
        ParamSave([W,V,U,s0],filename)
        best_TestPPL = currentTestPPL
        if s['verbose']:
            print 'NEW BEST: epoch', epoch, 'Test PPL', currentTestPPL, ' '*20
        s['BestEpoch'] = epoch
    else:
        print ''

    # learning rate decay if no improvement in 5 epochs
    if s['decay'] and abs(s['BestEpoch']-s['CurrentEpoch']) >= 2:
        s['learingRate'] *= 0.25
        print "Reducing Learning Rate to:", s['learingRate']
    if s['learingRate'] < 1e-5: break

print 'BEST RESULT: epoch', s['BestEpoch'], 'Test PPL', best_TestPPL