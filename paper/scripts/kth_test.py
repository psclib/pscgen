import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans as KMeans

from pscgen import NNU, Storage_Scheme
import utilities  as util

def bow(x, N):
    return np.bincount(x, minlength=N) / float(len(x))


data = np.load('/home/brad/data/kth_wang.npz')

X_tr = data['X_tr']
flat_X_tr = np.vstack(X_tr)
tr_subset = np.random.permutation(len(flat_X_tr))[:200000]
X_t = data['X_t']
ys_tr = data['Y_tr']
ys_t = data['Y_t']

eps = 0.95
alphas = [30]
nns_dists = []
pct_NN = {key: [] for key in alphas}
pct_approx_NN = {key: [] for key in alphas}


# # #nns
Ns = [1, 5, 10, 20, 50, 100, 245, 500, 750, 1000, 1500]
# Ns = [1, 2, 3, 4, 5]
for i, N in enumerate(Ns):
    D = KMeans(n_clusters=N)
    D.fit(flat_X_tr[tr_subset, 42:])
    D = D.cluster_centers_
    D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
    D_mean = np.mean(D, axis=0)
    D = D - D_mean

    svm_nns_xs_tr, svm_nns_xs_t = [], []
    for x in X_tr:
        x = x[:, 42:]
        x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
        x = x - D_mean
        nbrs = np.argmax(np.abs(np.dot(D, x.T)), axis=0)
        svm_nns_xs_tr.append(bow(nbrs, N))

    for x in X_t:
        x = x[:, 42:]
        x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
        x = x - D_mean
        nbrs = np.argmax(np.abs(np.dot(D, x.T)), axis=0)
        svm_nns_xs_t.append(bow(nbrs, N))

    acc = util.predict_chi2(svm_nns_xs_tr, ys_tr, svm_nns_xs_t, ys_t)
    print N, acc
    nns_dists.append(acc)

assert False


N = 750
D = np.loadtxt('/home/brad/data/D750_hog.csv', delimiter=',')
D = D.T

D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
D_mean = np.mean(D, axis=0)
D = D - D_mean

alphas = [1, 2, 3, 4, 5, 5, 5, 10, 10, 15, 25, 30]
betas = [1, 1, 1, 1, 1, 2, 4, 5, 10, 10, 25, 25]
storages = [Storage_Scheme.mini, Storage_Scheme.two_mini, Storage_Scheme.half]
storages = [Storage_Scheme.two_mini]

nnu_dists = {}
nnu_runtimes = {}
ABs = {}

pool = Pool(processes=4)

#NNU
for storage in storages:
    nnu = NNU(30, 25, storage)
    nnu.build_index(D)
    print nnu.name
    nnu_dists[nnu.name] = []
    nnu_runtimes[nnu.name] = []
    ABs[nnu.name] = []

    for alpha, beta in zip(alphas, betas):
        runtime_total = 0.0
        avg_abs = []
        svm_xs_tr, svm_xs_t = [], []
        total_matches = 0
        total_samples = 0


        svm_xs_tr = pool.map(enc_func, X_tr)
        svm_xs_t = pool.map(enc_func, X_t)
        # for i, x in enumerate(X_tr):
        #     total_samples += len(x)
        #     x = x[:, 42:42+96]
        #     nnu_nbrs, runtime, avg_ab = nnu.index(x, alpha=alpha, beta=beta, detail=True)
        #     svm_xs_tr.append(bow(nnu_nbrs, N))
        #     runtime_total += runtime
        #     avg_abs.append(avg_ab)
            
        # for x in X_t:
        #     total_samples += len(x)
        #     x = x[:, 42:42+96]
        #     nnu_nbrs, runtime, avg_ab = nnu.index(x, alpha=alpha, beta=beta, detail=True)
        #     svm_xs_t.append(bow(nnu_nbrs, N))
        #     runtime_total += runtime
        #     avg_abs.append(avg_ab)

        acc = util.predict_chi2(svm_xs_tr, ys_tr, svm_xs_t, ys_t)
        nnu_dists[nnu.name].append(acc)
        # nnu_runtimes[nnu.name].append(runtime_total)
        # ABs[nnu.name].append(np.mean(avg_abs))
        print alpha, beta, acc


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
