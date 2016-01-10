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
        print accuracy_score(Y[test_index], Y_pred)

    return acc / float(len(sss))



KMeans_tr_size = 200000
D_atoms = 500
ws = 100
X_flat, Y = util.wav_to_np('/home/brad/data/robot/')
X_window = []

for i, x in enumerate(X_flat):
    X_window.append(util.sliding_window(x, ws, 5))

X_Kmeans = np.vstack(X_window)[:KMeans_tr_size]
D = KMeans(n_clusters=D_atoms, init_size=D_atoms*3)
D.fit(X_Kmeans)

sss = StratifiedShuffleSplit(Y, 20, test_size=0.7, random_state=0)

dim_partitions = ['full', 'full', 'full', 'partition', 'partition']
dict_partitions = ['full', 'partition', 'partition', 'full', 'full']
num_nodes = [1, 5, 5, 5, 5]
num_dims = [1, 1, 1, 1, 1]
alphas = [10, 2, 5, 2, 5]
betas = [10, 2, 5, 2, 5]

items = zip(dict_partitions, dim_partitions, num_nodes, num_dims, alphas, betas)

for dict_partition, dim_partition, node, dim, alpha, beta in items[3:]:
        print dict_partition, dim_partition, node, dim
        nnuf = NNUForest(alpha=alpha, beta=beta, num_nodes=node, num_dims=dim,
                         dim_partition=dim_partition,
                         dict_partition=dict_partition)
        nnuf.build_index(D.cluster_centers_)
        print classify(nnuf, sss, X_window, Y)
        print ''
