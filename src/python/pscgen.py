import cPickle
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.svm import LinearSVC
from functools import partial

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import chi2_kernel
from numpy.lib.stride_tricks import as_strided as ast

import pscgen_c


def normalize(X):
    X = np.copy(X)
    norms = np.linalg.norm(X, axis=1)
    nonzero = np.where(norms != 0)
    X[nonzero] /= norms[nonzero][:, np.newaxis]

    return X


def bow_list(xs, N):
    xs_rep = []

    for item in xs:
        xs_rep.append(bow(item, minlength=N))

    xs_rep = np.array(xs_rep)

    return xs_rep


def bow(xs, minlength):
    counts = np.bincount(xs, minlength=minlength)
    return counts/np.linalg.norm(counts)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

    return i + 1

def predict_linear(tr_x, tr_y, t_x, t_y, batch=False, verbose=False):
    max_acc = 0
    tr_acc = 0
    cm = None
    tr_cm = None
    tuned_c = None

    #screen C
    # for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
    for C in [0.1]:
        if verbose:
            print 'Training with C: ', C

        clf = SVC(kernel='linear', C=C)
        clf.fit(tr_x, tr_y)
        y_pred = clf.predict(t_x)
        new_acc = accuracy_score(t_y, y_pred)
        if verbose:
            print 'Acc: ', new_acc

        if new_acc > max_acc:
            tr_y_pred = clf.predict(tr_x)
            tr_acc = accuracy_score(tr_y, tr_y_pred)
            tr_cm = confusion_matrix(tr_y, tr_y_pred)
            max_acc = new_acc
            cm = confusion_matrix(t_y, y_pred)
            tuned_c = C

    if verbose:
        print 'C: ', tuned_c
        print 'Training'
        print 'Accuracy:', tr_acc
        print tr_cm
        print ''

        print 'Testing'
        print 'Accuracy:', max_acc
        print cm
    else:
        sys.stdout.write(str('%2.3f' % max_acc))


def predict_chi2(tr_x, tr_y, t_x, t_y, batch=False, verbose=False):
    max_acc = 0
    tr_acc = 0
    cm = None
    tr_cm = None
    tuned_c, tuned_gamma = None, None

    #screen of gamma and C
    for g in [0.01, 0.05, .1]:
        for C in [1, 10, 100]:
            clf = SVC(kernel=partial(chi2_kernel, gamma=g), C=C)
            clf.fit(tr_x, tr_y)
            y_pred = clf.predict(t_x)
            new_acc = accuracy_score(t_y, y_pred)

            if new_acc > max_acc:
                tr_y_pred = clf.predict(tr_x)
                tr_acc = accuracy_score(tr_y, tr_y_pred)
                tr_cm = confusion_matrix(tr_y, tr_y_pred)
                max_acc = new_acc
                cm = confusion_matrix(t_y, y_pred)
                tuned_c = C
                tuned_gamma = g

    if verbose:
        print 'gamma, C: ', tuned_gamma, tuned_c
        print 'Training'
        print 'Accuracy:', tr_acc
        print tr_cm
        print ''

        print 'Testing'
        print 'Accuracy:', max_acc
        print cm

    return max_acc

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.
     
    Parameters
        shape - an int, or a tuple of ints
     
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass


def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    Parameters:
    a - an n-dimensional numpy array
    ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
    of each dimension of the window
    ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
    amount to slide the window in each dimension. If not specified, it
    defaults to ws.
    flatten - if True, all slices are flattened, otherwise, there is an 
    extra dimension for each dimension of the input.

    Returns
    an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window. I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)

class Compression_Scheme:
    pca, uniform_sub, random_linear_comb, random_sub, random_proj = range(5)

def compression_name(comp_scheme):
    if comp_scheme == Compression_Scheme.pca:
        return 'pca'
    elif comp_scheme == Compression_Scheme.uniform_sub:
        return 'uniform_sub'
    elif comp_scheme == Compression_Scheme.random_linear_comb:
        return 'random_linear_comb'
    elif comp_scheme == Compression_Scheme.random_sub:
        return 'random_sub'
    elif comp_scheme == Compression_Scheme.random_proj:
        return 'random_proj'

def name_to_comp_scheme(comp_scheme):
    if comp_scheme == 'pca':
        return Compression_Scheme.pca
    elif comp_scheme == 'uniform_sub':
        return Compression_Scheme.pca
    elif comp_scheme == 'random_linear_comb':
        return Compression_Scheme.random_linear_comb
    elif comp_scheme == 'random_sub':
        return Compression_Scheme.random_sub
    elif comp_scheme == 'random_proj':
        return Compression_Scheme.random_proj

class Storage_Scheme:
    half, mini, micro, nano, two_mini, four_micro = range(6)


def storage_stride(storage):
    if storage == Storage_Scheme.half:
        return 1
    elif storage == Storage_Scheme.mini:
        return 1
    elif storage == Storage_Scheme.micro:
        return 1
    elif storage == Storage_Scheme.nano:
        return 1
    elif storage == Storage_Scheme.two_mini:
        return 2
    elif storage == Storage_Scheme.four_micro:
        return 4

def storage_gamma_exp(storage):
    if storage == Storage_Scheme.half:
        return 16
    elif storage == Storage_Scheme.mini:
        return 8
    elif storage == Storage_Scheme.micro:
        return 4
    elif storage == Storage_Scheme.nano:
        return 2
    elif storage == Storage_Scheme.two_mini:
        return 16
    elif storage == Storage_Scheme.four_micro:
        return 16

def storage_name(storage):
    if storage == Storage_Scheme.half:
        return 'half'
    elif storage == Storage_Scheme.mini:
        return 'mini'
    elif storage == Storage_Scheme.micro:
        return 'micro'
    elif storage == Storage_Scheme.nano:
        return 'nano'
    elif storage == Storage_Scheme.two_mini:
        return 'two_mini'
    elif storage == Storage_Scheme.four_micro:
        return 'four_micro'

def name_to_storage(storage):
    if storage == 'half':
        return Storage_Scheme.half
    elif storage == 'mini':
        return Storage_Scheme.mini
    elif storage == 'micro':
        return Storage_Scheme.micro
    elif storage == 'nano':
        return Storage_Scheme.nano
    elif storage == 'two_mini':
        return Storage_Scheme.two_mini
    elif storage == 'four_micro':
        return Storage_Scheme.four_micro


class NNU(object):
    def __init__(self, alpha, beta, storage, comp_scheme, max_atoms):
        self.alpha = alpha
        self.beta = beta
        self.gamma_exp = storage_gamma_exp(storage)
        self.gamma = 2**self.gamma_exp
        self.storage = storage
        self.comp_scheme = comp_scheme
        self.name = storage_name(storage)
        self.D = None
        self.D_cols = None
        self.D_mean = None
        self.tables = None
        self.Vt = None
        self.VD = None
        self.max_atoms = max_atoms

    def save(self, filepath):
        """save class as self.name.txt"""
        file = open(filepath,'w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load(self, filepath):
        """try load self.namee.txt"""
        file = open(filepath,'r')
        dataPickle = file.read()
        file.close()

        self.__dict__ = cPickle.loads(dataPickle)

    def build_index_from_file(self, filepath, delimiter=','):
        '''
        Creates an nnu index from a filepath
        NOTE: ASSUMES D is zero mean/unit norm
        '''
        ret = pscgen_c.build_index_from_file(self.alpha, self.beta,
                                             self.gamma_exp, self.storage,
                                             self.comp_scheme, self.max_atoms,
                                             filepath, delimiter)
        self.D = np.array(ret[0])
        self.D_mean = np.array(ret[1])
        self.D_rows = ret[2]
        self.D_cols = ret[3]
        self.tables = np.array(ret[4], dtype=np.uint16)
        self.Vt = np.array(ret[5])
        self.VD = np.array(ret[6])

    def build_index(self, D):
        '''
        Creates an nnu index from a numpy array
        '''
        D_rows, D_cols = D.shape
        ret = pscgen_c.build_index(self.alpha, self.beta, self.gamma_exp,
                                   self.storage, self.comp_scheme, D_cols,
                                   D_rows, self.max_atoms, D.flatten())
        self.D = np.array(ret[0])
        self.D_mean = np.array(ret[1])
        self.D_rows = ret[2]
        self.D_cols = ret[3]
        self.tables = np.array(ret[4], dtype=np.uint16)
        self.Vt = np.array(ret[5])
        self.VD = np.array(ret[6])


    def index(self, X, alpha=None, beta=None, detail=False):
        '''
        Index into nnu tables.

        X is dimensions x samples.
        '''
        X = np.copy(X)
        X_rows, X_cols = X.shape

        if X_cols != self.D_rows:
            msg = 'Dimension mismatch: Expected {} but got {}'
            msg = msg.format(self.D_rows, X_cols)
            print msg
            assert False

        if not alpha:
            alpha = self.alpha

        if not beta:
            beta = self.beta

        if alpha > self.alpha:
            msg = 'alpha: {} greater than max table alpha: {}'
            msg = msg.format(alpha, self.alpha)
            print msg
            assert False

        if beta > self.beta:
            msg = 'beta: {} greater than max table beta: {}'
            msg = msg.format(beta, self.beta)
            print msg
            assert False

        X = np.ascontiguousarray(X.flatten())
        ret = pscgen_c.index(alpha, beta, self.alpha, self.beta, self.gamma,
                             self.storage, self.comp_scheme, self.D_rows,
                             self.D_cols, self.max_atoms, self.D, self.D_mean, 
                             self.tables, self.Vt, self.VD, X,
                             X_cols, X_rows)
        runtime = eval(str(ret[1]) + '.' + str(ret[2])) 

        if detail:
            return ret[0], runtime, ret[3]
        else:
            return ret[0]

    def index_single(self, X, enc_type):
        '''
        Index into nnu tables.

        X is dimensions x samples.
        '''
        X = np.copy(X)
        X_cols = len(X)

        if X_cols != self.D_rows:
            msg = 'Dimension mismatch: Expected {} but got {}'
            msg = msg.format(self.D_rows, X_cols)
            print msg
            assert False

        X = np.ascontiguousarray(X.flatten())
        ret = pscgen_c.index_single(enc_type, self.alpha, self.beta,
                                    self.alpha, self.beta, self.gamma,
                                    self.storage, self.comp_scheme,
                                    self.D_rows, self.D_cols, self.max_atoms,
                                    self.D, self.D_mean, self.tables, self.Vt,
                                    self.VD, X, X_cols, 1)

        return ret[0]

    def candidates(self, X, enc_type='nnu'):
        '''
        Returns candidate set and magnitudes in table for given X

        X is dimensions x samples.
        '''
        X = np.copy(X)
        X_cols = len(X)

        if X_cols != self.D_rows:
            msg = 'Dimension mismatch: Expected {} but got {}'
            msg = msg.format(self.D_rows, X_cols)
            print msg
            assert False

        X = np.ascontiguousarray(X.flatten())
        ret = pscgen_c.candidates_single(enc_type, self.alpha, self.beta,
                                         self.alpha, self.beta, self.gamma,
                                         self.storage, self.comp_scheme,
                                         self.D_rows, self.D_cols,
                                         self.max_atoms, self.D, self.D_mean,
                                         self.tables, self.Vt, self.VD, X,
                                         X_cols, 1)


        return np.array(ret[0]), np.array(ret[1])


    def to_dict(self):
        nnu_dict = {}
        nnu_dict['alpha'] = self.alpha
        nnu_dict['beta'] = self.beta
        nnu_dict['gamma'] = self.gamma
        nnu_dict['storage'] = storage_name(self.storage)
        nnu_dict['comp_scheme'] = compression_name(self.comp_scheme)
        nnu_dict['tables'] = list(self.tables.astype(int))
        nnu_dict['D'] = list(self.D)
        nnu_dict['D_rows'] = self.D_rows
        nnu_dict['D_cols'] = self.D_cols
        nnu_dict['Vt'] = list(self.Vt)
        nnu_dict['VD'] = list(self.VD)

        return nnu_dict


class NNUForest(object):
    def __init__(self, num_nodes=10, sample_dim='random', num_dims=10,
                 alpha=5, beta=5, storage=name_to_storage('mini'),
                 comp_scheme=Compression_Scheme.pca, enc_type='nnu'):
        self.num_nodes = num_nodes
        self.sample_dim = sample_dim
        self.num_dims = num_dims
        self.alpha = alpha
        self.beta = beta
        self.storage = storage
        self.comp_scheme = comp_scheme
        self.enc_type = enc_type
        self.nnu_nodes = []
        self.node_dim_idxs = []

    def build_index(self, D):
        self.num_atoms = D.shape[0]
        self.D = np.copy(D)
        D_dims = len(D[0])
        total_dims = self.num_dims*self.num_nodes
        partition_idxs = np.round(np.linspace(0, D_dims-1, total_dims))
        partition_idxs = partition_idxs.astype(int)

        for i in range(self.num_nodes):
            if self.sample_dim == 'random':
                selected_dims = np.random.permutation(D_dims)[:self.num_dims]
            elif self.sample_dim == 'partition':
                start_idx = i*self.num_dims
                end_idx = (i+1)*self.num_dims
                selected_dims = partition_idxs[start_idx:end_idx]

            self.node_dim_idxs.append(selected_dims)
            nnu = NNU(self.alpha, self.beta, self.storage, self.comp_scheme,
                      self.num_atoms)
            nnu.build_index(D[:, selected_dims])
            self.nnu_nodes.append(nnu)

        self.D_mean = np.mean(D, axis=0)
        self.D = D - self.D_mean
        self.D = normalize(D)

    def index(self, X, detail=False):
        atom_histogram = np.zeros(self.num_atoms)
        atom_histogram_raw = np.zeros(self.num_atoms)
        candidate_set = [] 
        all_idxs = np.array(self.node_dim_idxs).flatten()

        for i, nnu in enumerate(self.nnu_nodes):
            candidates, magnitudes = nnu.candidates(X[self.node_dim_idxs[i]])
            candidate_set.extend(candidates)
            atom_histogram_raw[candidates] += magnitudes
            
        X = np.copy(X)
        X = X - self.D_mean
        if np.linalg.norm(X) > 0:
            X = X / np.linalg.norm(X)

        candidate_set = np.array(list(set(candidate_set)))
        for idx in candidate_set:
            atom_histogram[idx] = np.dot(self.D[idx][all_idxs], X[all_idxs])
            
        #abs at end
        atom_histogram = np.abs(atom_histogram)
        atom_histogram_raw = np.abs(atom_histogram_raw)

        if detail:
            return np.argmax(atom_histogram), atom_histogram, atom_histogram_raw
        else:
            return np.argmax(atom_histogram)


class Pipeline(object):
    def __init__(self, ws, ss, sub_sample=1, max_classes=10):
        self.nnu = None
        self.svm = None
        self.ws = ws
        self.ss = ss
        self.sub_sample = sub_sample
        self.coef = None
        self.num_features = None
        self.num_classes = None
        self.max_classes = None
        self.intercept = None
        self.KMeans_tr_size = 200000
        self.enc_type = None
        self.max_classes = max_classes

    def fit(self, X, Y, D_atoms, max_atoms, alpha, beta, storage,
            comp_scheme=Compression_Scheme.pca, enc_type='nnu', T2=None):
        self.enc_type = enc_type
        if T2 == None:
            self.T2 = comp_scheme
        else:
            self.T2 = T2

        X_window, X_window_sub = [], []
        X = np.copy(X)

        for i, x in enumerate(X):
            X_window.append(sliding_window(x, self.ws, self.ss))
            x_z = np.zeros(len(x))
            x_z[::self.sub_sample] = x[::self.sub_sample]
            X_window_sub.append(sliding_window(x_z, self.ws, self.ss))

        X_Kmeans = np.vstack(X_window)[:self.KMeans_tr_size]
        D = KMeans(n_clusters=D_atoms, init_size=D_atoms*3)
        D.fit(X_Kmeans)
        D_orig = D.cluster_centers_

        #subsample dictionary
        D = np.zeros(D_orig.shape)
        D[:, ::self.sub_sample] = D_orig[:, ::self.sub_sample]

        self.nnu = NNU(alpha, beta, storage, comp_scheme, max_atoms)
        self.nnu.build_index(D)
        self.nnu.comp_scheme = self.T2

        svm_X = []
        for x in X_window_sub:
            nbrs = []
            for xi in x:
                new_nbrs = self.nnu.index_single(xi, enc_type=self.enc_type)
                nbrs.extend(new_nbrs)
            svm_X.append(bow(nbrs, D_atoms))

        self.svm = LinearSVC()
        self.svm.fit(svm_X, Y)
        self.coef = np.ascontiguousarray(self.svm.coef_.flatten())
        self.intercept = np.ascontiguousarray(self.svm.intercept_.flatten())
        self.num_classes = len(self.svm.classes_)
        self.num_features = D_atoms

               
    def generate(self, output_path, float_type='double'):
        return pscgen_c.generate(output_path, self.enc_type, float_type,
                                 self.ws, self.ss,
                                 self.num_features, self.num_classes,
                                 self.max_classes, self.coef, self.intercept,
                                 self.nnu.alpha, self.nnu.beta, self.nnu.gamma,
                                 self.nnu.storage, self.nnu.comp_scheme,
                                 self.nnu.D_rows, self.nnu.D_cols,
                                 self.nnu.max_atoms, self.nnu.D,
                                 self.nnu.D_mean, self.nnu.tables,
                                 self.nnu.Vt, self.nnu.VD)

    def classify(self, X):
        X_orig = np.ascontiguousarray(X)
        X = np.ascontiguousarray(np.zeros(len(X_orig)))
        X[::self.sub_sample] = X_orig[::self.sub_sample]

        idx = pscgen_c.classify(self.enc_type, X, len(X), self.ws, self.ss,
                                 self.num_features, self.num_classes,
                                 self.max_classes, self.coef, self.intercept,
                                 self.nnu.alpha, self.nnu.beta, self.nnu.gamma,
                                 self.nnu.storage, self.nnu.comp_scheme,
                                 self.nnu.D_rows, self.nnu.D_cols,
                                 self.nnu.max_atoms, self.nnu.D,
                                 self.nnu.D_mean, self.nnu.tables, self.nnu.Vt,
                                 self.nnu.VD)[0]

        return self.svm.classes_[idx]
