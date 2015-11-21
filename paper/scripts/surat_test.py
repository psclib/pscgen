import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans as KMeans
from multiprocessing import Pool
from functools import partial
import random

from pscgen import NNU, Storage_Scheme
import utilities  as util
np.random.seed(42)


def bow(x, N):
    return np.bincount(x, minlength=N) / float(len(x))

def nnu_to_bow(x, alpha, beta, N, nnu):
    x = x[:, 42:42+96]
    nbrs = nnu.index(x, alpha=alpha, beta=beta)
    return bow(nbrs, N)

data = np.load('/home/brad/data/surat_voice.npz')
X = data['X']
Y = data['Y']
XY = zip(X, Y)
random.shuffle(XY)
X, Y = zip(*XY)
tr_size = int(len(X)*0.8)

X_tr = X[:tr_size]
X_t = X[tr_size:]
Y_tr = Y[:tr_size]
Y_t = Y[tr_size:]
X_tr_Kmeans = np.vstack(X_tr)[:200000]

nns_dists = []

# # # #nns
# Ns = [1, 5, 10, 20, 50, 100, 245, 500, 750, 1500, 3000, 5000]
# # Ns = [1, 2, 3, 4, 5]
# for i, N in enumerate(Ns):
#     D = KMeans(n_clusters=N)
#     D.fit(X_tr_Kmeans)
#     D = D.cluster_centers_
#     D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
#     D_mean = np.mean(D, axis=0)
#     D = D - D_mean

#     svm_nns_xs_tr, svm_nns_xs_t = [], []
#     for x in X_tr:
#         x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
#         x = x - D_mean
#         nbrs = np.argmax(np.abs(np.dot(D, x.T)), axis=0)
#         svm_nns_xs_tr.append(bow(nbrs, N))

#     for x in X_t:
#         x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
#         x = x - D_mean
#         nbrs = np.argmax(np.abs(np.dot(D, x.T)), axis=0)
#         svm_nns_xs_t.append(bow(nbrs, N))

#     acc = util.predict_chi2(svm_nns_xs_tr, Y_tr, svm_nns_xs_t, Y_t)
#     print N, acc
#     nns_dists.append(acc)

# assert False

N = 250
D = KMeans(n_clusters=N)
D.fit(X_tr_Kmeans)
D = D.cluster_centers_

# D = np.loadtxt('/home/brad/data/D750_hog.csv', delimiter=',')
# D = D.T

D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
D_mean = np.mean(D, axis=0)
D = D - D_mean

alphas = [1, 2, 3, 4, 5, 5, 5, 10, 10, 15, 25, 30]
betas = [1, 1, 1, 1, 1, 2, 4, 5, 10, 10, 25, 25]
storages = [Storage_Scheme.mini, Storage_Scheme.two_mini, Storage_Scheme.half]
storages = [Storage_Scheme.mini]

nnu_dists = {}
nnu_runtimes = {}
ABs = {}


#NNU
for storage in storages:
    nnu = NNU(5, 5, storage)
    print nnu.name
    nnu.build_index(D)
    nnu_dists[nnu.name] = []
    nnu_runtimes[nnu.name] = []
    ABs[nnu.name] = []

    for alpha, beta in zip(alphas, betas):
        runtime_total = 0.0
        avg_abs = []
        svm_xs_tr, svm_xs_t = [], []
        total_matches = 0
        total_samples = 0

        enc_func = partial(nnu_to_bow, alpha=alpha, beta=beta, nnu=nnu, N=N)

        for i, x in enumerate(X_tr):
            print i
            total_samples += len(x)
            nnu_nbrs, runtime, avg_ab = nnu.index(x, alpha=alpha, beta=beta, detail=True)
            svm_xs_tr.append(bow(nnu_nbrs, N))
            runtime_total += runtime
            avg_abs.append(avg_ab)
            
        for x in X_t:
            total_samples += len(x)
            nnu_nbrs, runtime, avg_ab = nnu.index(x, alpha=alpha, beta=beta, detail=True)
            svm_xs_t.append(bow(nnu_nbrs, N))
            runtime_total += runtime
            avg_abs.append(avg_ab)

        acc = util.predict_chi2(svm_xs_tr, Y_tr, svm_xs_t, Y_t)
        nnu_dists[nnu.name].append(acc)
        # nnu_runtimes[nnu.name].append(runtime_total)
        # ABs[nnu.name].append(np.mean(avg_abs))
        print alpha, beta, acc


assert False
# plt.plot(Ns, nns_dists, '-', label='nns', linewidth=2)

names = ['half', 'mini', 'two_mini']
labels = ['Single(16)', 'Single(8)', 'Combined(8 + 8)']
for name, label in zip(names, labels):
    plt.plot(Ns, nnu_dists[name], label='nnu', linewidth=2)

plt.xlabel('Number of Dot Products')
plt.ylabel('Classification Accuracy')
plt.legend(loc='lower right')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.15/accuracy.png')
plt.clf()


# #plot 1
# plt.plot(Ns, nns_dists, '-', label='nns')
# for alpha in alphas:
#     plt.plot(Ns, nnu_dists[alpha], '-', label='nnu({}, 10)'.format(alpha))
# plt.xlabel('Number of D atoms')
# plt.ylabel('Accuracy')
# plt.legend(loc='upper left')
# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
# plt.savefig('/home/brad/acc.png')
# plt.clf()


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
