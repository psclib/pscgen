import cPickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.svm import SVC

import utilities as util
import pscgen_c

def normalize(X):
    X = np.copy(X)
    norms = np.linalg.norm(X, axis=1)
    nonzero = np.where(norms != 0)
    X[nonzero] /= norms[nonzero][:, np.newaxis]

    return X


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
    def __init__(self, alpha, beta, storage):
        self.alpha = alpha
        self.beta = beta
        self.gamma_exp = storage_gamma_exp(storage)
        self.gamma = 2**self.gamma_exp
        self.storage = storage
        self.name = storage_name(storage)
        self.D = None
        self.D_cols = None
        self.D_mean = None
        self.tables = None
        self.Vt = None
        self.VD = None

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
                                             filepath, delimiter)
        self.D = np.array(ret[0])
        self.D_rows = ret[1]
        self.D_cols = ret[2]
        self.tables = np.array(ret[3], dtype=np.uint16)
        self.Vt = np.array(ret[4])
        self.VD = np.array(ret[5])

    def build_index(self, D):
        '''
        Creates an nnu index from a numpy array
        '''
        #normalize D
        D = normalize(D)

        #subtract mean
        self.D_mean = np.mean(D, axis=0)
        D = D - self.D_mean

        D_rows, D_cols = D.shape
        ret = pscgen_c.build_index(self.alpha, self.beta, self.gamma_exp,
                                   self.storage, D.flatten(), D_cols, D_rows)
        self.D = np.array(ret[0])
        self.D_rows = ret[1]
        self.D_cols = ret[2]
        self.tables = np.array(ret[3], dtype=np.uint16)
        self.Vt = np.array(ret[4])
        self.VD = np.array(ret[5])


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

        #normalize X
        X = normalize(X)

        #subtract D mean
        X = X - self.D_mean

        X = np.ascontiguousarray(X.flatten())
        ret = pscgen_c.index(alpha, beta, self.alpha, self.beta, self.gamma,
                             self.storage, self.D, self.D_rows, self.D_cols,
                             self.D_mean, self.tables, self.Vt, self.VD, X,
                             X_cols, X_rows)
        runtime = eval(str(ret[1]) + '.' + str(ret[2])) 

        if detail:
            return ret[0], runtime, ret[3]
        else:
            return ret[0]

    def to_dict(self):
        nnu_dict = {}
        nnu_dict['alpha'] = self.alpha
        nnu_dict['beta'] = self.beta
        nnu_dict['gamma'] = self.gamma
        nnu_dict['storage'] = storage_name(self.storage)
        nnu_dict['tables'] = list(self.tables.astype(int))
        nnu_dict['D'] = list(self.D)
        nnu_dict['D_rows'] = self.D_rows
        nnu_dict['D_cols'] = self.D_cols
        nnu_dict['Vt'] = list(self.Vt)
        nnu_dict['VD'] = list(self.VD)

        return nnu_dict

class Pipeline(object):
    def __init__(self):
        self.nnu = None
        self.svm = None
        self.KMeans_tr_size = 200000

    def fit(self, X, Y, D_atoms, alpha, beta, storage):
        X_Kmeans = np.vstack(X)[:self.KMeans_tr_size]
        D = KMeans(n_clusters=D_atoms, init_size=D_atoms*3)
        D.fit(X_Kmeans)
        D = D.cluster_centers_
        D = util.normalize(D)
        D_mean = np.mean(D, axis=0)
        D = D - D_mean

        svm_X = []
        for x in X:
            x = util.normalize(x)
            x = x - D_mean
            nbrs = np.argmax(np.abs(np.dot(D, x.T)), axis=0)
            svm_X.append(util.bow(nbrs, D_atoms))

        self.svm = SVC(kernel='linear')
        self.svm.fit(svm_X, Y)

        self.nnu = NNU(alpha, beta, storage)
        self.nnu.build_index(D)
       

    def classify(self, X):
        return pscgen_c.classify(X, self.nnu, self.svm)
