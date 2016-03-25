import pickle

import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

import utilities as util
from pscgen import NNUForest, NNUDictDim, normalize

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


def num_kops(N, M):
    return (N * (2 * M - 1)) / 1000.

def candidate_match(nnuf, x, D, D_mean):
    match_total = 0
    total = 0
    num_candidates = 0
    for xi in x:
        idx, mag, num_c = nnuf.index(xi, detail=True)
        num_candidates += num_c

        if np.linalg.norm(xi) > 0:
            xi = xi / np.linalg.norm(xi)

        xi = xi - D_mean

        max_idx = np.argmax(np.abs(np.dot(D, xi)))

        if max_idx == idx:
            match_total += 1

        total += 1

    return match_total, total, num_candidates


def pct_found(nnuf, X, D, D_mean):
    ret_tup = Parallel(n_jobs=8)(delayed(candidate_match)(nnuf, x, D, D_mean)
                                                          for x in X)
    match_totals, totals, num_candidates = zip(*ret_tup)
    match_total = np.sum(match_totals)
    total = np.sum(totals)
    num_candidates = np.sum(num_candidates)

    return match_total / float(total), num_candidates / float(total)


def classify(nnuf, X, Y, train_index, test_index):
    svm_X = []
    num_candidates = 0
    for i in train_index:
        nbrs = []
        for xi in X[i]:
            idx, mag, num_c = nnuf.index(xi, detail=True)
            num_candidates += num_c
            nbrs.append(idx)
        svm_X.append(util.bow(nbrs, D_atoms))

    svm = LinearSVC()
    svm.fit(svm_X, Y[train_index])

    svm_X = []
    for i in test_index:
        nbrs = []
        for xi in X[i]:
            idx, mag, num_c = nnuf.index(xi, detail=True)
            num_candidates += num_c
            nbrs.append(idx)
        svm_X.append(util.bow(nbrs, D_atoms))

    Y_pred = svm.predict(svm_X)

    return accuracy_score(Y[test_index], Y_pred), num_candidates
    

def accuracy_test(names, num_l1_nodes, alphas, betas, dict_partition, dim_partition, 
                   X, Y, D, D_mean, sss, model='NNUF', num_l2_nodes=0):
    plt_accs, plt_ops = {}, {}
    plt_ops = []
    plt_accs = []
    for alpha, beta in zip(alphas, betas):
        print alpha, beta
        if model == 'NNUF':
            nnuf = NNUForest(alpha=alpha, beta=beta, num_nodes=num_l1_nodes, num_dims=dim,
                             dim_partition=dim_partition,
                             dict_partition=dict_partition)
        elif model == 'DictDim':
            nnuf = NNUDictDim(num_l1_nodes, num_l2_nodes, alpha, beta)

        nnuf.build_index(D)
        ret_tup = Parallel(n_jobs=8, verbose=0)(delayed(classify)(nnuf, X_window, Y,
                                                    train_index, test_index)
                                           for train_index, test_index in sss)
        accs, num_candidates = zip(*ret_tup)
        avg_candidates = np.sum(num_candidates) / float(num_X*num_folds)
        plt_accs.append(np.mean(accs))
        plt_ops.append(num_kops(avg_candidates, len(X[0][0])))

    return plt_ops, plt_accs


def candidate_test(names, num_nodes, alphas, betas, dict_partition,
                   dim_partition, X, D, D_mean):

    plt_accs, plt_ops = {}, {}
    for name, node in zip(names, num_nodes):
        plt_ops[name] = []
        plt_accs[name] = []
        for alpha, beta in zip(alphas, betas):
            nnuf = NNUForest(alpha=alpha, beta=beta, num_nodes=node,
                             num_dims=dim, dim_partition=dim_partition,
                             dict_partition=dict_partition)
            nnuf.build_index(D)
            pct, avg_candidates = pct_found(nnuf, X, D, D_mean)
            plt_accs[name].append(pct)
            plt_ops[name].append(num_kops(avg_candidates, len(X[0][0])))


    return plt_ops, plt_accs




fig_name = 'forest_acc'
KMeans_tr_size = 200000
D_atoms = 500
ws = 50
subsample_pcts = [1., 1., 1., 0.5]
num_l1_nodes = [4, 4, 2, 2]
num_l2_nodes = [0, 0, 2, 2]
names = ['Full', 'DicP', 'DicP+DimP', 'DicP+DimP+Sub']
models = ['NNUF', 'NNUF', 'DictDim', 'DictDim']
num_dims = [1, 1, 1, 1, 1]
alphas = [1, 3, 5, 5, 5, 10, 10, 15, 20]
betas =  [1, 1, 1, 2, 5,  5,  7, 10, 12]

dim = 1
dict_partitions = ['full', 'partition', '', '']
dim_partitions = ['full', 'full', '', '']

X_flat, Y = util.wav_to_np('/home/brad/data/robot/')
X = []

num_folds = 5
sss = StratifiedShuffleSplit(Y, num_folds, test_size=0.7, random_state=0)


for i, x in enumerate(X_flat):
    X.append(util.sliding_window(x, ws, 5))


plt_dict = {}

for name, num_l1_node, num_l2_node, dict_partition, dim_partition, subsample_pct, model in \
zip(names, num_l1_nodes, num_l2_nodes, dict_partitions, dim_partitions, subsample_pcts, models):
    print name
    if name == 'Full':
        run_alpha = alphas
        run_beta = betas
    else:
        run_alpha = alphas[:5]
        run_beta = betas[:5]

    sample_dims = np.linspace(0, ws-1, int(subsample_pct*ws)).astype(int)
    # sample_dims = np.random.permutation(ws)[:int(subsample_pct*ws)]
    X_window = [x[:, sample_dims] for x in X]

    X_Kmeans = np.vstack(X_window)[:KMeans_tr_size]
    num_X = len(np.vstack(X_window))
    D = KMeans(n_clusters=D_atoms, init_size=D_atoms*3)
    D.fit(X_Kmeans)
    D = D.cluster_centers_
    D = normalize(D)
    D_mean = np.mean(D, axis=0)
    D = D - D_mean


    plt_ops, plt_accs = accuracy_test(names, num_l1_node, alphas, betas,
                                      dict_partition, dim_partition, 
                                      X_window, Y, D, D_mean, sss, model,
                                      num_l2_node)
    # plt_ops, plt_accs = candidate_test(names, num_nodes, alphas, betas,
    #                                    dict_partition, dim_partition, X_window,
    #                                    D, D_mean)


    plt_dict[name] = (plt_ops, plt_accs)
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 

for name in names:
    plt_ops, plt_accs = plt_dict[name]
    ax.plot(plt_ops, plt_accs, linewidth=linewidth, label=name)

plt.xlabel('Number of Arithmetic Operations (Kops)')
plt.ylabel('Classification Accuracy')
# plt.ylabel('Percent of True Nearest Neighbor Found')
ax.set_ylim((0, 1.0))
ax.set_xlim((0, plt_dict['Full'][0][-1]))
plt.legend(loc='lower right')
fig = plt.gcf()
plt.savefig('../new/' + fig_name + '.pdf')
plt.clf()
pickle.dump(plt_dict, open("../new/save.p", "wb") ) 
