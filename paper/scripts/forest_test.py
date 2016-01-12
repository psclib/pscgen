import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

import utilities as util
from pscgen import NNUForest, normalize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'axes.labelsize': 12,
   'text.fontsize': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [10, 10]})

linewidth = 2

def num_ops(nodes, alpha, beta, dims):
    return nodes * alpha * beta * dims


def pct_found(nnuf, X, D, D_mean):
    match_total = 0
    total = 0
    num_candidates = 0
    for x in X:
        for xi in x:
            idx, num_c = nnuf.index(xi, detail=True)
            num_candidates += num_c
            if np.linalg.norm(xi) > 0:
                xi = xi / np.linalg.norm(xi)

            xi = xi - D_mean
   
            max_idx = np.argmax(np.abs(np.dot(D, xi)))
            if max_idx == idx:
                match_total += 1

            total += 1

    return match_total / float(total), num_candidates / float(total)


def classify(nnuf, X, Y, train_index, test_index):
    svm_X = []
    num_candidates = 0
    for i in train_index:
        nbrs = []
        for xi in X[i]:
            idx, num_c = nnuf.index(xi, detail=True)
            nbrs.append(idx)
        svm_X.append(util.bow(nbrs, D_atoms))

    svm = LinearSVC()
    svm.fit(svm_X, Y[train_index])

    svm_X = []
    for i in test_index:
        nbrs = []
        for xi in X[i]:
            idx, num_c = nnuf.index(xi, detail=True)
            num_candidates += num_c
            nbrs.append(idx)
        svm_X.append(util.bow(nbrs, D_atoms))

    Y_pred = svm.predict(svm_X)

    return accuracy_score(Y[test_index], Y_pred), num_candidates
    

KMeans_tr_size = 200000
D_atoms = 500
ws = 100
X_flat, Y = util.wav_to_np('/home/brad/data/robot/')
X_window = []

for i, x in enumerate(X_flat):
    X_window.append(util.sliding_window(x, ws, 5))

X_Kmeans = np.vstack(X_window)[:KMeans_tr_size]
num_X = len(np.vstack(X_window))
D = KMeans(n_clusters=D_atoms, init_size=D_atoms*3)
D.fit(X_Kmeans)
D = D.cluster_centers_
D = normalize(D)
D_mean = np.mean(D, axis=0)
D = D - D_mean


num_folds = 8
sss = StratifiedShuffleSplit(Y, num_folds, test_size=0.7, random_state=0)

num_nodes = [1, 3, 5]
names = ['NNU', 'DictPart(3)', 'DictPart(5)']
num_dims = [1, 1, 1, 1, 1]
alphas = [1, 2, 3, 4, 5, 5, 5, 5, 5, 10, 10, 10, 15, 20, 20]
betas = [1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 7, 10, 10, 10, 12]

dim = 1
dict_partition = 'full'
dim_partition = 'partition'

plt_ops, plt_accs = {}, {}

for name, node in zip(names, num_nodes):
    plt_ops[name] = []
    plt_accs[name] = []
    for alpha, beta in zip(alphas, betas):
        print name, node, alpha, beta
        nnuf = NNUForest(alpha=alpha, beta=beta, num_nodes=node, num_dims=dim,
                         dim_partition=dim_partition,
                         dict_partition=dict_partition)
        nnuf.build_index(D)
        # pct, avg_candidates = pct_found(nnuf, X_window, D, D_mean)
        # print alpha, beta, pct
        ret_tup = Parallel(n_jobs=8, verbose=5)(delayed(classify)(nnuf, X_window, Y,
                                                    train_index, test_index)
                                           for train_index, test_index in sss)
        accs, num_candidates = zip(*ret_tup)
        avg_candidates = np.sum(num_candidates) / float(num_X*num_folds)

        # print num_candidates
        # print np.mean(accs), avg_candidates
        plt_accs[name].append(np.mean(accs))
        plt_ops[name].append(avg_candidates)



fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
for name in names:
    ax.plot(plt_ops[name], plt_accs[name], linewidth=linewidth, label=name)

plt.xlabel('Average Number of Candidates')
# plt.ylabel('Classification Accuracy')
plt.ylabel('Percent of Candidates Found')
ax.set_ylim(ymin=0)
plt.legend(loc='lower right')
fig = plt.gcf()
plt.savefig('../figures/forest_acc_dim.pdf')
plt.clf()


assert False
num_nodes = [1, 3, 5]
names = ['NNU', 'DimPart(3)', 'DimPart(5)']
dim = 1
dict_partition = 'full'
dim_partition = 'partition'

plt_ops, plt_accs = {}, {}

for name, node in zip(names, num_nodes):
    plt_ops[name] = []
    plt_accs[name] = []
    for alpha, beta in zip(alphas, betas):
        print name, node, alpha, beta
        nnuf = NNUForest(alpha=alpha, beta=beta, num_nodes=node, num_dims=dim,
                         dim_partition=dim_partition,
                         dict_partition=dict_partition)
        nnuf.build_index(D.cluster_centers_)
        ret_tup = Parallel(n_jobs=8, verbose=5)(delayed(classify)(nnuf, X_window, Y,
                                                    train_index, test_index)
                                           for train_index, test_index in sss)
        accs, num_candidates = zip(*ret_tup)
        avg_candidates = np.sum(num_candidates) / float(num_X*num_folds)

        print num_candidates
        print np.mean(accs), avg_candidates
        plt_accs[name].append(np.mean(accs))
        plt_ops[name].append(avg_candidates)


fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
for name in names:
    ax.plot(plt_ops[name], plt_accs[name], linewidth=linewidth, label=name)

plt.xlabel('Average Number of Candidates')
plt.ylabel('Classification Accuracy')
ax.set_ylim(ymin=0)
plt.legend(loc='lower right')
fig = plt.gcf()
plt.savefig('../figures/forest_acc_dim.pdf')
plt.clf()
