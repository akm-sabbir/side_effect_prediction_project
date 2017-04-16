import json
import glob
import xml.etree.ElementTree as ET
import re
import random
import math
from scipy.sparse import lil_matrix
from theano import tensor as T
import csv
#import pickle
import cPickle as pickle
import random
import sys
import theano.sandbox.cuda
#theano.sandbox.cuda.use(sys.argv[2])
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from theano.ifelse import ifelse
from sklearn import cross_validation
from sklearn.utils import shuffle
import theano
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
write_file = sys.argv[1]

srng = RandomStreams(seed=902340)
np.random.seed(82342)

def normal(shape, scale=0.05):
    return np.asarray(np.random.randn(*shape) * scale).astype('float32')

def uniform(shape, scale=0.05):
        return np.random.uniform(low=-scale, high=scale, size=shape).astype('float32')

def as_floatX(variable):
    if isinstance(variable, float):
        #return np.cast[theano.config.floatX](variable)
        return np.cast['float32'](variable)

    if isinstance(variable, np.ndarray):
        #return np.cast[theano.config.floatX](variable)
        return np.cast['float32'](variable)
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def rectify(X):
    return T.maximum(X, 0.)

def elu(X):
    return T.switch(T.ge(X, 0), X, T.exp(X)-1.)

def dropout(X, dropout_switch=1, p=0.):
    retain_prob = 1 - p
    mask = srng.binomial(X.shape, p=retain_prob, dtype='float32')
    X = ifelse(T.lt(dropout_switch, 0.5), X*mask, (X*retain_prob).reshape(mask.shape))
    return X


def sgd(cost, parameters, lr, updates=None):

    grads = T.grad(cost,parameters)
    updates = {}
    for param,grad in zip(parameters,grads):
            updates[param] = param - lr*grad

    return updates

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def adagrad(cost, params, updates=None, max_norm=9.0,
                                lr=0.001, eps=1e-6, rho=0.95):
    lr = theano.shared(np.float32(lr).astype(theano.config.floatX))
    eps = np.float32(eps).astype(theano.config.floatX)
    rho = theano.shared(np.float32(rho).astype(theano.config.floatX))
    if max_norm is not None and max_norm is not False:
        max_norm = theano.shared(np.float32(max_norm).astype(theano.config.floatX))

    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in params]
    xsums   = [None for param in params]

    gparams = T.grad(cost, params)

    if updates is None:
        updates = OrderedDict()

    for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):
        # clip gradients if they get too big
        if max_norm is not None and max_norm is not False and (param.name!='Words'):
            grad_norm = gparam.norm(L=2)
            gparam = (T.minimum(max_norm, grad_norm)/ (grad_norm + eps)) * gparam

        updates[gsum] =  T.cast(gsum + (gparam ** 2), theano.config.floatX)
        updates[param] =  T.cast(param - lr * (gparam / (T.sqrt(updates[gsum] + eps))), theano.config.floatX)

    #return updates, gsums, xsums, lr, max_norm
    return updates



def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        #if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words') and (param.name!='Location') and (param.name!='POS') and (param.name!='DEPREL'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 
############ predictive model works on any dataset#############
### just need to format the data###########################
class Model(object):

    #def __init__(self, data_x_word, data_x_loc_e1, data_x_loc_e2, data_x_pos, ne, npos, nloc,
    def __init__(self, emb, sent_size, ne,
            nf=100, nc=2, de=300, p_drop=0.5, fs=[3,4,5]):
        '''
            data_x_word :: Training word indices 
            ne :: Number of words
            nf :: Number of feature maps
            nc :: Number of classes
            de :: Dimensionality of word embeddings
            p_drop :: Dropout probability
            fs :: Convolution filter widths
        '''
        batch_size = 5
        test_size = 1
        self.emb = theano.shared(name='Words',
            value=np.asarray(emb, dtype='float32'))
        self.filter_w = []
        self.filter_b = []
        for filter_size in fs:
            w_bound = np.sqrt(2./((de)*filter_size*nf))
            print nf, filter_size, de, de
            self.filter_w.append(theano.shared(
                value=normal((nf, 1, filter_size, de), w_bound)
                .astype('float32')))
            self.filter_b.append(theano.shared(
                value=np.zeros((nf,)).astype('float32')))
        w_o_bound = np.sqrt(2./(nf*len(fs)))
        self.w_o = theano.shared(
                value=normal((nf*len(fs), nc), w_o_bound)
                .astype('float32'))
        self.b_o = theano.shared(
            value=np.zeros((nc,)).astype('float32'))


        zero_vec_tensor = T.cast(T.vector(), 'float32')
        self.zero_vec = np.zeros(de).astype('float32')
        self.set_zero = theano.function([zero_vec_tensor], updates=[(self.emb, T.set_subtensor(self.emb[0,:], zero_vec_tensor))])
        self.params = [self.w_o, self.b_o, self.emb]
        for w, b in zip(self.filter_w, self.filter_b):
            self.params.append(w)
            self.params.append(b)

        dropout_switch = T.fscalar('dropout_switch')
        idxs = T.matrix()
        x_word = self.emb[T.cast(idxs.flatten(), 'int32')].reshape((idxs.shape[0], 1, idxs.shape[1], de))
        y = T.ivector('y')

        l1_w_trains = []
        for w, b, width in zip(self.filter_w, self.filter_b, fs):
            l1_w_train = conv2d(x_word, w, image_shape=(batch_size,1,None,de), filter_shape=(nf, 1, width, de))
            l1_w_train = rectify(l1_w_train + b.dimshuffle('x', 0, 'x', 'x'))
            #l1_w_train = max_pool_2d(l1_w_train, (sent_size-width+1, 1)).flatten(2)
            l1_w_train = T.max(l1_w_train, axis=2).flatten(2)
            l1_w_trains.append(l1_w_train)

        l1_w_tests = []
        for w, b, width in zip(self.filter_w, self.filter_b, fs):
            l1_w_test = conv2d(x_word, w, image_shape=(test_size,1,sent_size,de), filter_shape=(nf, 1, width, de))
            l1_w_test = rectify(l1_w_test + b.dimshuffle('x', 0, 'x', 'x'))
            #l1_w_test = max_pool_2d(l1_w_test, (sent_size-width+1, 1)).flatten(2)
            l1_w_test = T.max(l1_w_test, axis=2).flatten(2)
            l1_w_tests.append(l1_w_test)

        l1_w_tests_small = []
        for w, b, width in zip(self.filter_w, self.filter_b, fs):
            l1_w_test_small = conv2d(x_word, w, image_shape=(1,1,sent_size,de), filter_shape=(nf, 1, width, de))
            l1_w_test_small = rectify(l1_w_test_small + b.dimshuffle('x', 0, 'x', 'x'))
            #l1_w_test_small = max_pool_2d(l1_w_test_small, (sent_size-width+1, 1)).flatten(2)
            l1_w_test_small = T.max(l1_w_test_small, axis=2).flatten(2)
            l1_w_tests_small.append(l1_w_test_small)

        l1_train = T.concatenate(l1_w_trains, axis=1)
        l1_train = dropout(l1_train, dropout_switch, p_drop)

        l1_test = T.concatenate(l1_w_tests, axis=1)
        l1_test = dropout(l1_test, dropout_switch, p_drop)
        l1_test_small = T.concatenate(l1_w_tests_small, axis=1)
        l1_test_small = dropout(l1_test_small, dropout_switch, p_drop)


        p_y_given_x_train = T.nnet.softmax(T.dot(l1_train, self.w_o) + self.b_o)
        p_y_given_x = T.nnet.softmax(T.dot(l1_test, self.w_o) + self.b_o)
        p_y_given_x_small = T.nnet.softmax(T.dot(l1_test_small, self.w_o) + self.b_o)
        y_pred = T.argmax(p_y_given_x, axis=1)
        y_pred_small = T.argmax(p_y_given_x_small, axis=1)
        index = T.ivector()
        learning_rate = T.scalar()

        negative_log_liklihood = -T.mean(T.log(p_y_given_x_train)[T.arange(y.shape[0]), y])
        updates = sgd_updates_adadelta(self.params, negative_log_liklihood, 0.95, 1e-6, 9)

        self.train_model2 = theano.function([idxs, y, dropout_switch], negative_log_liklihood, updates = updates)
        self.f = theano.function([idxs, dropout_switch], l1_train)

        self.predict = theano.function(inputs=[idxs, dropout_switch], outputs=y_pred, allow_input_downcast = True)

        self.predict_small = theano.function(inputs=[idxs, dropout_switch], outputs=y_pred_small, allow_input_downcast = True)
        self.predict_proba_small = theano.function(inputs = [idxs, dropout_switch], outputs = p_y_given_x_small, allow_input_downcast = True)

    def __getstate__(self):
        data = [self.emb.get_value(), self.w_o.get_value(), self.b_o.get_value()]
        data += [x.get_value() for x in self.filter_w]
        data += [x.get_value() for x in self.filter_b]
        return data

    def set_word_emb(self, X):
        self.emb.set_value(X)

class Tokenizer(object):
    def __init__(self):
        self.pattern = re.compile(r'(?u)\b\w\w+\b')
        self.words = {}
        self.data = []
        self.total = 0

    def fit_transform(self, dataset):
        data = []
        token_list = []
        self.total = 1
        count = {}
        count2 = {}
        for line in dataset:
            tokens = re.findall(self.pattern, line.rstrip().lower())
            tmp = []
            for t in tokens:
                if t not in self.words:
                    self.words[t] = self.total
                    tmp.append(self.total)
                    count[t] = 1
                    self.total += 1
                else:
                    count[t] += 1
            for t in set(tokens):
                if t not in count2:
                    count2[t] = 1
                else:
                    count2[t] += 1

        #tmp = [k for k,v in count.iteritems() if v > 5 and v < 100000 and len(k) < 20 and count2[k] < 10000 and not k.isdigit() and len(k) > 2]
        tmp = [k for k,v in count.iteritems() if v > 5]
        #tmp = [k for k,v in count.iteritems()]
        self.words = dict(zip(tmp, range(1,len(tmp)+1)))
        first = True
        line_num = 0
        big_cnt = 0
        for line in dataset:
            line_num += 1
            tokens = re.findall(self.pattern, line.rstrip().lower())
            test = []
            tmp = []
            for t in tokens:
                if t in self.words:
                    tmp.append(self.words[t])
                    test.append(t)
            if len(tokens) >= 3000:
                big_cnt += 1

            token_list.append(tmp[:10000])
        print 'big_cnt',big_cnt
        sys.stdout.flush()

        return token_list

    def transform(self, dataset):
        data = []
        token_list = []
        for line in dataset:
            tokens = re.findall(self.pattern, line.rstrip().lower())
            tmp = []
            for t in tokens:
                if t not in self.words:
                    continue
                else:
                    tmp.append(self.words[t])
            print len(tmp)
            sys.stdout.flush()
            token_list.append(list(tmp))
        return token_list

def mae(true, pred):
    level2score = {'ABSENT': 0, 'MILD': 1, 'MODERATE': 2, 'SEVERE': 3}
    total = 0
    ae = 0
    level_cnts = {'ABSENT': 0, 'MILD': 0, 'MODERATE': 0, 'SEVERE': 0}
    aes = []
    for c in ['ABSENT', 'MILD', 'MODERATE', 'SEVERE']:
        c_cnt = 0
        c_ae = 0
        for t,p in zip(true, pred):
            if t != c:
                continue
            c_cnt += 1
            c_ae += abs(level2score[p] - level2score[t])
        if level2score[c] in (0,3):
            ae += 100. * (1 - (c_ae)/(float(c_cnt)*3.))
            aes.append(100. * (1 - (c_ae)/(float(c_cnt)*3.)))
        else:
            ae += 100. * (1 - (c_ae)/(float(c_cnt)*2.))
            aes.append(100. * (1 - (c_ae)/(float(c_cnt)*2.)))
    return ae/4., aes

def main():
    level2score = {'ABSENT': 0, 'MILD': 1, 'MODERATE': 2, 'SEVERE': 3}
    score2level = {0:'ABSENT', 1:'MILD', 2:'MODERATE', 3:'SEVERE'}
    data = []
    y = []
    print 'loading data'
    sys.stdout.flush()
    files = []
    #with open('data_FT_non_cancer.json') as in_file:
    for filename in glob.glob('./*.xml'):
        source = ET.parse(filename)
        root = source.getroot()
        for c in root:
            if c.tag == 'TEXT':
                data.append(c.text)
        score = source.findall('./TAGS/POSITIVE_VALENCE')[0].attrib['score']
        score = score.upper().strip()
        y.append(int(level2score[score]))
        files.append(filename)
    for filename in glob.glob('./annotated_by_one/*.xml'):
        source = ET.parse(filename)
        root = source.getroot()
        for c in root:
            if c.tag == 'TEXT':
                data.append(c.text)
        score = source.findall('./TAGS/POSITIVE_VALENCE')[0].attrib['score']
        score = score.upper().strip()
        y.append(int(level2score[score]))
        files.append(filename)


    for i in y:
        print i
    print 'vectorizing data'
    sys.stdout.flush()
    vec = Tokenizer()
    #with open('tokenizer.pkl', 'rb') as in_file:
    #    vec = pickle.load(in_file)
    corpus = vec.fit_transform(data)
    corpus = vec.transform(data)
    print vec.total
    print corpus[0]
    max_len = np.max([len(x) for x in corpus])+1
    mean_len = np.mean([len(x) for x in corpus])+1
    m = np.max([np.max(x+[0]) for x in corpus])
    train_word_inds = []
    zeros = [0,0,0,0,0]
    print 'padding data', max_len, mean_len, m
    sys.stdout.flush()
    max_len_list = []
    for d in corpus:
        padding = [0]*(max_len - len(list(d)))
        max_len_list.append(len(list(d))-1)
        train_word_inds.append(list(zeros)+list(d)+list(padding)+list(zeros))
        #print max_len, len(list(d)), len(padding), len(list(zeros)+list(d)+list(padding)+list(zeros))

    train_word_inds = np.array(train_word_inds)

    y_train = np.array(y)

    print 'Creating Model'
    sys.stdout.flush()
    num_feat_maps = 300
    word_dim = 128 # 300
    drop_p = 0.5
    num_classes = 4 # dynamic
    filter_widths = [3,4,5]
    num_words = np.max(train_word_inds)+1
    embs = np.random.normal(0, 1./word_dim, size=(num_words, word_dim))
    #with open('rel_vecs.pkl', 'rb') as in_file:
    #    embs  = pickle.load(in_file)
    num_words = embs.shape[0]
    print embs.shape, 'emb shape'
    #sss = StratifiedShuffleSplit(y, 1, test_size=0.1, random_state=0)
    cv = StratifiedKFold(y_train, n_folds = 2, random_state = 1234)
    print 'Model Created'
    sys.stdout.flush()

    cv_cnt = 1
    for train_index, test_index in cv:
        print 'CV1'
        sys.stdout.flush()
        split_y_train2, spltit_y_test = np.array(y_train[train_index], dtype='int32'), np.array(y_train[test_index], dtype='int32')
        split_train_word_inds2, split_test_word_inds = np.array(train_word_inds[train_index], dtype='float32'), np.array(train_word_inds[test_index], dtype='float32')
        sys.stdout.flush()
        sss = ShuffleSplit(split_y_train2.shape[0], 1, test_size = 0.1)
        final_test_preds = None
        for train_index_l2, val_index in sss:
            print 'CV2'
            sys.stdout.flush()
            split_train_word_inds, split_val_word_inds = np.array(split_train_word_inds2[train_index_l2], dtype='float32'), np.array(split_train_word_inds2[val_index], dtype='float32')
            split_y_train, split_y_val = np.array(split_y_train2[train_index_l2], dtype='int32'), np.array(split_y_train2[val_index], dtype='int32')

            mod = Model(embs, split_train_word_inds.shape[1], num_words,
                num_feat_maps, num_classes, word_dim, drop_p, filter_widths)
            embs = np.random.normal(0, 1./word_dim, size = (num_words, word_dim))

            num_epochs = 2
            best_val = 0
            best_tps = 0
            best_score = -9999
            cur_test_preds = None
            for epoch in range(1, num_epochs+1):
                batches = list(range(split_train_word_inds.shape[0]))
                random.shuffle(batches)
                total_cost = 0
                num_batches = 0
                tot_batches = len(batches)
                for start, end in zip(range(0, split_train_word_inds.shape[0], 5), range(5, split_train_word_inds.shape[0], 5)):
                    #cost = mod.f(split_train_word_inds[start:end], 0)
                    #print cost.shape
                    check_len = np.max(max_len_list[start:end])
                    samples = np.random.choice(np.arange(split_train_word_inds.shape[1]), replace= False, size = split_train_word_inds.shape[1]/2.).flatten().tolist()
                    samples.sort()
                    sample_inds = [x for x in samples if x < check_len]
                    #cost = mod.train_model2(split_train_word_inds[start:end, sample_inds], split_y_train[start:end], 0)
                    cost = mod.train_model2(split_train_word_inds[batches[start:end]].astype('float32'), split_y_train[batches[start:end]].astype('int32'), 0)
                    total_cost += cost
                    mod.set_zero(mod.zero_vec)
                    num_batches += 1
                    print 'cv', cv_cnt, 'epoch', epoch, 'batch', num_batches, tot_batches/5, check_len, len(sample_inds)
                    sys.stdout.flush()

                val_y_pred = []
                for start, end in zip(range(0, split_val_word_inds.shape[0]), range(1, split_val_word_inds.shape[0]+1)):
                    pred = mod.predict_small(split_val_word_inds[start:end], 1)
                    val_y_pred.append(pred.flatten()[0])
                mae_score = mae([score2level[x] for x in  split_y_val], [score2level[x] for x in val_y_pred])
                print 'EPOCH:',epoch, 'cv', cv_cnt, 'validation', epoch, mae_score, best_score
                if mae_score[0] > best_score:
                    best_score = mae_score[0]
                    test_y_pred = []
                    
                    test_y_proba = []
                    for start, end in zip(range(0, split_test_word_inds.shape[0]), range(1, split_test_word_inds.shape[0]+1)):
                        pred = mod.predict_small(split_test_word_inds[start:end], 1)
                        test_y_pred.append(pred.flatten()[0])
                        pred = mod.predict_proba_small(split_test_word_inds[start:end], 1)
                        test_y_proba.append(pred.flatten())
                    cur_test_preds = np.array(test_y_proba)
                    test_mae_score = mae([score2level[x] for x in  split_y_test], [score2level[x] for x in test_y_pred])
                    print 'TESTING EPOCH', epoch, 'CV', cv_cnt, test_mae_score
                #with open('save_data_bin_multi_his/'+str(cv_cnt)+'_cv_his.pkl', 'wb') as out_file:
                #    data = mod.__getstate__()
                #    pickle.dump(data, out_file)
            if final_test_preds is None:
                final_test_preds = cur_test_preds.copy()
            else:
                final_test_preds = final_test_preds.copy() + cur_test_preds.copy()
            print 'MAX', final_test_preds.max()
            del mod
        preds = np.argmax(final_test_preds, axis=1)
        test_mae_score = mae([score2level[x] for x in  split_y_test], [score2level[x] for x in preds])
        print 'TEST FINAL cv', cv_cnt, test_mae_score
        cv_cnt += 1
main()

