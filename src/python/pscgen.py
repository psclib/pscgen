import cPickle
import numpy as np
import libpypscgen as pypscgen

class NNU(object):
    def __init__(self, alpha, beta, gamma_exp):
        self.alpha = alpha
        self.beta = beta
        self.gamma_exp = gamma_exp
        self.gamma = 2**gamma_exp
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
                                             self.gamma_exp, filepath,
                                             delimiter)
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
                                   D.flatten(), D_cols, D_rows)
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
                             self.D, self.D_rows, self.D_cols, self.tables,
                             self.Vt, self.VD, X, X_cols, X_rows)
        runtime = eval(str(ret[1]) + '.' + str(ret[2])) 

        if detail:
            return ret[0], runtime, ret[3]
        else:
            return ret[0]
