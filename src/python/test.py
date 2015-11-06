import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans as KMeans

import pscgen

X_tr = np.loadtxt('/home/brad/data/notredame/medium.csv', delimiter=',')
# X_t = np.loadtxt('/home/brad/data/notredame/full.csv', delimiter=',')
X_t = X_tr[:, :5000]
X_tr = X_tr[:, 5000:]

print X_t.shape, X_tr.shape
X_tr = X_tr.T
X_t = X_t.T
X_tr = X_tr / np.linalg.norm(X_tr, axis=1)[:, np.newaxis]
X_t = X_t / np.linalg.norm(X_t, axis=1)[:, np.newaxis]
print 'data loaded'

eps = 0.95
alphas = [5, 10, 20, 30]
nns_dists = {key: [] for key in alphas}
nnu_dists = {key: [] for key in alphas}
nnu_runtimes = {key: [] for key in alphas}
nnu_D_diffs = {key: [] for key in alphas}
ABs = {key: [] for key in alphas}
pct_NN = {key: [] for key in alphas}
pct_approx_NN = {key: [] for key in alphas}

Ns = [50, 100, 200, 500, 1000, 1500]
# Ns = [10, 20, 30, 40, 50]
for i, N in enumerate(Ns):
    D = KMeans(n_clusters=N)
    D.fit(X_tr)
    D = D.cluster_centers_
    # D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
    nbrs = np.argmax(np.dot(D, X_t.T), axis=0)
    np.savetxt('D.csv', D.T, delimiter=',', fmt='%2.6f')

    for alpha in alphas:
        print N, alpha
        nnu = pscgen.NNU(alpha, 10)
        nnu.build_index('D.csv')

        nnu_nbrs, runtime, avg_ab = nnu.index(X_t.T)
        nnu_runtimes[alpha].append(runtime)
        nnu_nbrs = np.array(nnu_nbrs).astype(int)
        ABs[alpha].append(avg_ab)

        diff = (nnu_nbrs - nbrs)
        diff[diff > 1] = 1
        diff[diff < 0] = 1

        pct =  1 - (np.sum(diff) / float(len(diff)))
        pct_NN[alpha].append(pct)

        dists = np.einsum('ij,ij->j', X_t.T, D[nbrs].T)
        nns_dists[alpha].append(np.mean(dists))

        dists = np.einsum('ij,ij->j', X_t.T, D[nnu_nbrs].T)
        nnu_dists[alpha].append(np.mean(dists))

        D_diff = np.einsum('ij,ij->j', D[nbrs].T, D[nnu_nbrs].T)
        approx_pct = len(np.where(D_diff > eps)[0]) / float(len(D_diff))
        pct_approx_NN[alpha].append(approx_pct)
        

#plot 1
plt.plot(Ns, nns_dists[alphas[0]], '-', label='nns')
for alpha in alphas:
    plt.plot(Ns, nnu_dists[alpha], '-', label='nnu({}, 10)'.format(alpha))
plt.xlabel('Number of D atoms')
plt.ylabel('Mean cosine similarity between test samples and chosen D')
plt.legend()
plt.savefig('/home/brad/1.png')
plt.clf()


for alpha in alphas:
    plt.plot(Ns, pct_NN[alpha], '-', label='exact({})'.format(alpha))
    plt.plot(Ns, pct_approx_NN[alpha], '-', label='similar({})'.format(alpha))
plt.xlabel('Number of D atoms')
plt.ylabel('% NN found')
plt.legend()
plt.savefig('/home/brad/2.png')
plt.clf()


for alpha in alphas:
    plt.plot(Ns, nnu_runtimes[alpha], '-', label='nnu({})'.format(alpha))
plt.xlabel('Number of D atoms')
plt.ylabel('runtime (seconds)')
plt.legend()
plt.savefig('/home/brad/3.png')
plt.clf()
    
for alpha in alphas:
    plt.plot(Ns, ABs[alpha], '-', label='nnu({})'.format(alpha))
plt.xlabel('Number of D atoms')
plt.ylabel('Average Candidate set size')
plt.legend()
plt.savefig('/home/brad/4.png')
plt.clf()
