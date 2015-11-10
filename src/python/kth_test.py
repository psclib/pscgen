import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans as KMeans

import pscgen
import utilities  as util
np.random.seed(42)

def bow(x, N):
    return np.bincount(x, minlength=N) / float(len(x))


data = np.load('/home/brad/data/kth_wang.npz')

X_tr = data['xs_tr']
flat_X_tr = np.vstack(X_tr)
tr_subset = np.random.permutation(len(flat_X_tr))[:200000]
X_t = data['xs_t']
ys_tr = data['ys_tr']
ys_t = data['ys_t']

eps = 0.95
# alphas = [5, 10, 20, 30]
alphas = [30]
nns_dists = []
# nnu_dists = {key: [] for key in alphas}
# nnu_runtimes = {key: [] for key in alphas}
# nnu_D_diffs = {key: [] for key in alphas}
# ABs = {key: [] for key in alphas}
pct_NN = {key: [] for key in alphas}
pct_approx_NN = {key: [] for key in alphas}

Ns = [1, 5, 10, 20, 50, 100, 240, 500, 750]
for i, N in enumerate(Ns):
    D = KMeans(n_clusters=N)
    D.fit(flat_X_tr[tr_subset])
    D = D.cluster_centers_
    D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
    np.savetxt('D.csv', D.T, delimiter=',', fmt='%2.6f')

    #NNS
    svm_nns_xs_tr, svm_nns_xs_t = [], []
    for x in X_tr:
        nbrs = np.argmax(np.dot(D, x.T), axis=0)
        svm_nns_xs_tr.append(bow(nbrs, N))

    for x in X_t:
        nbrs = np.argmax(np.dot(D, x.T), axis=0)
        svm_nns_xs_t.append(bow(nbrs, N))

    acc = util.predict_chi2(svm_nns_xs_tr, ys_tr, svm_nns_xs_t, ys_t)
    nns_dists.append(acc)


N = 1500
D = KMeans(n_clusters=N)
D.fit(flat_X_tr[tr_subset])
D = D.cluster_centers_
D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
np.savetxt('D.csv', D.T, delimiter=',', fmt='%2.6f')


alphas = [1, 5, 5, 10, 10, 20, 20, 25, 30]
betas = [1, 1, 2, 2, 5, 5, 12, 25, 25]

nnu_dists = []
nnu_runtimes = []
ABs = []

#NNU
for alpha, beta in zip(alphas, betas):
    print N, alpha
    nnu = pscgen.NNU(alpha, beta)
    nnu.build_index_from_file('D.csv')
    runtime_total = 0.0
    avg_abs = []
    svm_xs_tr, svm_xs_t = [], []

    print 'Enc training'
    for i, x in enumerate(X_tr):
        # np.savetxt('x.csv', x.T, fmt='%2.6f')
        # nbrs = np.argmax(np.dot(D, x.T), axis=0)
        nnu_nbrs, runtime, avg_ab = nnu.index(x.T)
        svm_xs_tr.append(bow(nnu_nbrs, N))
        runtime_total += runtime
        avg_abs.append(avg_ab)
        
    print 'Enc testing'
    for x in X_t:
        # nbrs = np.argmax(np.dot(D, x.T), axis=0)
        nnu_nbrs, runtime, avg_ab = nnu.index(x.T)
        svm_xs_t.append(bow(nnu_nbrs, N))
        runtime_total += runtime
        avg_abs.append(avg_ab)

    print 'SVM'
    acc = util.predict_chi2(svm_xs_tr, ys_tr, svm_xs_t, ys_t)
    nnu_dists.append(acc)
    nnu_runtimes.append(runtime_total)
    ABs.append(np.mean(avg_abs))
    print acc, runtime_total


plt.plot(Ns, nns_dists, '-', label='nns')
plt.plot(Ns, nnu_dists, '-', label='nnu')
plt.xlabel('Number of dot products')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/dot.png')
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
