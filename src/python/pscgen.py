import cPickle
import libpypscgen as pypscgen

class NNU(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.D = None
        self.D_rows = None
        self.D_cols = None
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

    def build_index(self, filepath, delimiter=','):
        '''
        Creates an nnu index from a filepath
        '''
        ret = pypscgen.new_dict(self.alpha, self.beta, filepath, delimiter)
        self.D = ret[0]
        self.D_rows = ret[1]
        self.D_cols = ret[2]
        self.tables = ret[3]
        self.Vt = ret[4]
        self.VD = ret[5]

    def index(self, X):
        '''
        Index into nnu tables.

        X is dimensions x samples.
        '''
        X_rows, X_cols = X.shape
        X = X.T.flatten()
        return pypscgen.encode(self.alpha, self.beta, self.D, self.D_rows,
                               self.D_cols, self.tables, self.Vt, self.VD,
                               X, X_rows, X_cols)
