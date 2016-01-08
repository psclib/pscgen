'''Attempt to find a projection better than PCA'''

import sys
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.decomposition import PCA
from sklearn import linear_model

import utilities as util

def match_pct(a, b):
    return len(np.where(a == b)[0]) / float(len(a))


N = 25
M = 10000
KMeans_tr_size = 200000
D_atoms = 500
zoom_dim = 20
data_folder = sys.argv[1] 
output_folder = sys.argv[2]

if output_folder[-1] != '/':
    output_folder += '/'

X, Y = util.wav_to_np(data_folder)
X = [util.sliding_window(x, 40, 20) for x in X]

X = np.vstack(X)
X = X[np.random.permutation(len(X))]
X_Kmeans = X[:KMeans_tr_size]
D = KMeans(n_clusters=D_atoms, init_size=D_atoms*3)
D.fit(X)
D = D.cluster_centers_

D = util.normalize(D)
X = util.normalize(X)
D_mean = np.mean(D, axis=0)
D = D - D_mean
X = X - D_mean
U, S_D, V = np.linalg.svd(D)
_, S_X, _ = np.linalg.svd(X[:M])
V = V.T
# V = np.random.randn(*V.shape)
V = V / np.linalg.norm(V, axis=1)
VD = np.dot(V, D.T).T
VX = np.dot(V, X.T).T

true_idxs = np.argmax(np.abs(np.dot(X, D.T)), axis=1)

for i in range(10):
    v1_dists = cdist(VX[:, i][:, np.newaxis], VD[:, i][:,np.newaxis])
    v1_idxs = np.argmin(np.abs(v1_dists), axis=1)
    pca_match_pct = match_pct(true_idxs, v1_idxs)
    print pca_match_pct


print  '---'

C = X - D[true_idxs]

C_tr = C[np.random.permutation(len(C))[:10000]]
# pca = PCA()
# pca.fit(C_tr)
U, S_D, V = np.linalg.svd(C_tr)


# VX = np.dot(pca.components_, X.T).T
# VD = np.dot(pca.components_, D.T).T

VX = np.dot(V, X.T).T
VD = np.dot(V, D.T).T

for i in range(10):
    v1_dists = cdist(VX[:, i][:, np.newaxis], VD[:, i][:,np.newaxis])
    v1_idxs = np.argmin(np.abs(v1_dists), axis=1)
    pca_match_pct = match_pct(true_idxs, v1_idxs)
    print pca_match_pct
