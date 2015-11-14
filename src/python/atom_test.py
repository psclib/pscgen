import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans

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
nnu_dists = {key: [] for key in alphas}
nnu_runtimes = {key: [] for key in alphas}
nnu_D_diffs = {key: [] for key in alphas}
ABs = {key: [] for key in alphas}
pct_NN = {key: [] for key in alphas}
pct_approx_NN = {key: [] for key in alphas}

Ns = [18, 23, 30, 37, 45, 52]
for i, N in enumerate(Ns):
    D = KMeans(n_clusters=N)
    D.fit(flat_X_tr[tr_subset])
    D = D.cluster_centers_
    D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
    # np.savetxt('D.csv', D.T, delimiter=',', fmt='%2.6f')

    #NNS
    svm_nns_xs_tr, svm_nns_xs_t = [], []
    for x in X_tr:
        nbrs = np.argmax(np.dot(D, x.T), axis=0)
        svm_nns_xs_tr.append(bow(nbrs, N))

    for x in X_t:
        nbrs = np.argmax(np.dot(D, x.T), axis=0)
        svm_nns_xs_t.append(bow(nbrs, N))

    acc = util.predict_chi2(svm_nns_xs_tr, ys_tr, svm_nns_xs_t, ys_t)
    print acc
    nns_dists.append(acc)
