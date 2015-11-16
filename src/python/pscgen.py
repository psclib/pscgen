import cPickle
import numpy as np
import libpypscgen as pypscgen

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


class NNU(object):
    def __init__(self, alpha, beta, storage):
        self.alpha = alpha
        self.beta = beta
        self.gamma_exp = storage_gamma_exp(storage)
        self.gamma = 2**self.gamma_exp
        self.storage = storage
        self.name = storage_name(storage)
        self.D = None
        self.D_rows = None
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
        ret = pypscgen.build_index_from_file(self.alpha, self.beta,
                                             self.gamma_exp, self.storage,
                                             filepath, delimiter)
        self.D = ret[0]
        self.D_rows = ret[1]
        self.D_cols = ret[2]
        self.tables = ret[3]
        self.Vt = ret[4]
        self.VD = ret[5]

    def build_index(self, D):
        '''
        Creates an nnu index from a numpy array
        '''

        #normalize D
        D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]

        #subtract mean
        self.D_mean = np.mean(D, axis=0)
        D = D - self.D_mean

        D_rows, D_cols = D.shape
        ret = pypscgen.build_index(self.alpha, self.beta, self.gamma_exp,
                                   self.storage, D.flatten(), D_cols, D_rows)
        self.D = ret[0]
        self.D_rows = ret[1]
        self.D_cols = ret[2]
        self.tables = ret[3]
        self.Vt = ret[4]
        self.VD = ret[5]


    def index(self, X, alpha=None, beta=None, detail=False):
        '''
        Index into nnu tables.

        X is dimensions x samples.
        '''
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
        X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]

        #subtract mean
        X = X - self.D_mean

        X = np.ascontiguousarray(X.flatten())
        ret = pypscgen.index(alpha, beta, self.alpha, self.beta, self.gamma,
                             self.storage, self.D, self.D_rows, self.D_cols,
                             self.tables, self.Vt, self.VD, X, X_cols, X_rows)
        runtime = eval(str(ret[1]) + '.' + str(ret[2])) 

        if detail:
            return ret[0], runtime, ret[3]
        else:
            return ret[0]
