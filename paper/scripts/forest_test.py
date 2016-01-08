import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import utilities as util
from pscgen import NNUForest


def classify(nnuf, sss, X_window, Y):
    acc = 0.0
    for train_index, test_index in sss:
        svm_X = []
        for i in train_index:
            nbrs = []
            for xi in X_window[i]:
                idx = nnuf.index(xi)
                nbrs.append(idx)
            svm_X.append(util.bow(nbrs, D_atoms))

        svm = LinearSVC()
        svm.fit(svm_X, Y[train_index])

        svm_X = []
        for i in test_index:
            nbrs = []
            for xi in X_window[i]:
                idx = nnuf.index(xi)
                nbrs.append(idx)
            svm_X.append(util.bow(nbrs, D_atoms))

        Y_pred = svm.predict(svm_X)
        acc += accuracy_score(Y[test_index], Y_pred)

    return acc / float(len(sss))



KMeans_tr_size = 200000
D_atoms = 400
ws = 200
X_flat, Y = util.wav_to_np('/home/brad/data/robot/')
X_window = []

for i, x in enumerate(X_flat):
    X_window.append(util.sliding_window(x, ws, 5))

X_Kmeans = np.vstack(X_window)[:KMeans_tr_size]
D = KMeans(n_clusters=D_atoms, init_size=D_atoms*3)
D.fit(X_Kmeans)

sss = StratifiedShuffleSplit(Y, 5, test_size=0.7, random_state=0)

sample_dims = ['partition', 'random']
num_nodes = [5]
num_dims = [10]

for sample_dim in sample_dims:
    for node, dim in zip(num_nodes, num_dims):
        print sample_dim, node, dim
        nnuf = NNUForest(alpha=10, beta=10, num_nodes=node, num_dims=dim,
                         sample_dim=sample_dim)
        nnuf.build_index(D.cluster_centers_)
        print classify(nnuf, sss, X_window, Y)
        print ''
