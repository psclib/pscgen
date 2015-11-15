import numpy as np
import matplotlib.pyplot as plt

from pscgen import NNU


D = np.loadtxt('/home/brad/data/D1500_hog.csv', delimiter=',')
X = np.loadtxt('/home/brad/data/kth_test_hog.csv', delimiter=',')

#transpose to be (samples, features)
D = D.T
X = X.T

#copy
NNU_D = np.copy(D)
NNU_X = np.copy(X)

nnu = NNU(15, 25, 16)
nnu.build_index(NNU_D)

#python analysis
X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
X = X - nnu.D_mean

D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
D = D - nnu.D_mean

nbrs = np.argmax(np.dot(D, X.T), axis=0)



alphas = [1, 5, 10, 15]
betas = [1, 5, 10, 20, 25]
for alpha in alphas:
    for beta in betas:
        nnu_nbrs = nnu.index(NNU_X, alpha=alpha, beta=beta)
        print len(np.where(nnu_nbrs == nbrs)[0]) / float(len(X))
