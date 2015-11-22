# Runs comparison for nns/nnu at different settings
import sys
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans as KMeans

from pscgen import NNU, Storage_Scheme
import utilities as util

def read_dataset(data_path, tr_pct=0.8):
    '''The data format has either keys 'X' and 'Y' or
       keys 'X_tr', 'X_t', 'Y_tr', 'Y_t' for datasets
       that specify the training and testing fold
    '''
    data = np.load(data_path)

    if 'X' in data.keys():
        X = data['X']
        Y = data['Y']
        XY = zip(X, Y)
        random.shuffle(XY)
        X, Y = zip(*XY)
        tr_size = int(len(X)*tr_pct)

        X_tr = X[:tr_size]
        X_t = X[tr_size:]
        Y_tr = Y[:tr_size]
        Y_t = Y[tr_size:]
    else:
        X_tr = data['X_tr']
        X_t = data['X_t']
        Y_tr = data['Y_tr']
        Y_t = data['Y_t']

    return X_tr, X_t, Y_tr, Y_t

data_path = sys.argv[1]
fig_name = sys.argv[2]
NNU_N = 750
KMeans_tr_size = 200000
alphas = [1, 2, 3, 4, 5, 5, 5, 10, 10, 15, 15, 30]
betas = [1, 1, 1, 1, 1, 2, 4, 5, 10, 10, 15, 25]
Ns = [1, 2, 3, 4, 5, 10, 20, 50, 100, 245, 500, 750]
nns_dists = []
storages = [Storage_Scheme.half, Storage_Scheme.two_mini, Storage_Scheme.mini]
nnu_dists = {}
nnu_runtimes = {}
ABs = {}
X_tr, X_t, Y_tr, Y_t = read_dataset(data_path)
X_tr_Kmeans = np.vstack(X_tr)[:KMeans_tr_size]

#nns
print 'Nearest Neighbor'
for i, N in enumerate(Ns):
    D = KMeans(n_clusters=N, init_size=N*3)
    D.fit(X_tr_Kmeans)
    D = D.cluster_centers_
    D = util.normalize(D)
    D_mean = np.mean(D, axis=0)
    D = D - D_mean

    svm_nns_xs_tr, svm_nns_xs_t = [], []
    for x in X_tr:
        x = util.normalize(x)
        x = x - D_mean
        nbrs = np.argmax(np.abs(np.dot(D, x.T)), axis=0)
        svm_nns_xs_tr.append(util.bow(nbrs, N))

    for x in X_t:
        x = util.normalize(x)
        x = x - D_mean
        nbrs = np.argmax(np.abs(np.dot(D, x.T)), axis=0)
        svm_nns_xs_t.append(util.bow(nbrs, N))

    acc = util.predict_chi2(svm_nns_xs_tr, Y_tr, svm_nns_xs_t, Y_t)
    print N, acc
    nns_dists.append(acc)


D = KMeans(n_clusters=NNU_N, init_size=NNU_N*3)
D.fit(X_tr_Kmeans)
D = D.cluster_centers_

#NNU
for storage in storages:
    nnu = NNU(max(alphas), max(betas), storage)
    print 'NNU: ' + nnu.name

    nnu.build_index(D)
    nnu_dists[nnu.name] = []
    nnu_runtimes[nnu.name] = []
    ABs[nnu.name] = []

    for alpha, beta in zip(alphas, betas):
        runtime_total = 0.0
        avg_abs = []
        svm_xs_tr, svm_xs_t = [], []

        for i, x in enumerate(X_tr):
            nnu_nbrs, runtime, avg_ab = nnu.index(x, alpha=alpha, beta=beta,
                                                  detail=True)
            svm_xs_tr.append(util.bow(nnu_nbrs, NNU_N))
            runtime_total += runtime
            avg_abs.append(avg_ab)
            
        for x in X_t:
            nnu_nbrs, runtime, avg_ab = nnu.index(x, alpha=alpha, beta=beta,
                                                  detail=True)
            svm_xs_t.append(util.bow(nnu_nbrs, NNU_N))
            runtime_total += runtime
            avg_abs.append(avg_ab)

        acc = util.predict_chi2(svm_xs_tr, Y_tr, svm_xs_t, Y_t)
        nnu_dists[nnu.name].append(acc)
        nnu_runtimes[nnu.name].append(runtime_total)
        ABs[nnu.name].append(np.mean(avg_abs))
        print alpha, beta, acc


names = ['half', 'mini', 'two_mini']
labels = ['NNU - 16 bit', 'NNU - 8 bit', 'NNU - (8 bit + 8 bit)']

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.plot(Ns, nns_dists, label='Nearest Neighbor', linewidth=2)
for name, label in zip(names, labels):
    ax.plot(Ns, nnu_dists[name], label=label, linewidth=2)

ax.set_xscale('log')
plt.xlabel('Number of Dot Products')
plt.ylabel('Classification Accuracy')
plt.legend(loc='lower right')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('../figures/' + fig_name + '_accuracy.png')
plt.clf()


# for alpha in alphas:
#     plt.plot(Ns, nnu_runtimes[alpha], '-', label='nnu({})'.format(alpha))
# plt.xlabel('Number of D atoms')
# plt.ylabel('runtime (seconds)')
# plt.legend(loc='upper left')
# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
# plt.savefig('/home/brad/runtime.png')
# plt.clf()
    
# for alpha in alphas:
#     plt.plot(Ns, ABs[alpha], '-', label='nnu({})'.format(alpha))
# plt.xlabel('Number of D atoms')
# plt.ylabel('Average Candidate set size')
# plt.legend(loc='upper left')
# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
# plt.savefig('/home/brad/avg_candidates.png')
# plt.clf()
