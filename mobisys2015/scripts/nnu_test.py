import numpy as np
import sys

from pscgen import NNU, Storage_Scheme

def gen_colormap(max_alpha, max_beta, storage, NNU_X, NNU_D,
                 nbrs, base_path, random=False, verbose=False):
    base_path = '/home/brad/11.15/'
    if random:
        iden = 'random'
    else:
        nnu = NNU(max_alpha, max_beta, storage)
        nnu.build_index(NNU_D)
        iden = nnu.name

    count = np.zeros((max_alpha, max_beta))
    for alpha in range(1, max_alpha+1):
        for beta in range(1, max_beta+1):
            if not random:
                nnu_nbrs = nnu.index(NNU_X, alpha=alpha, beta=beta)
            else:
                nnu_nbrs = []
                for x in X:
                    idxs = np.random.permutation(len(D))[:alpha*beta]
                    nnu_nbrs.append(idxs[np.argmax(np.abs(np.dot(D[idxs], x)))])
            
            pct_found = len(np.where(nnu_nbrs == nbrs)[0]) / float(len(X))
            count[alpha-1, beta-1] = pct_found
            print alpha, beta, count[alpha-1, beta-1]

    # plt.imshow(count, interpolation='nearest', vmin=0, vmax=1)
    # plt.colorbar()
    # fig = plt.gcf()
    # fig.set_size_inches(18.5, 10.5)
    # plt.savefig(base_path + iden + 'color.png')
    # plt.clf()
    np.save(base_path + iden + '_count', count)


D = np.loadtxt('/home/brad/data/D1500_hog.csv', delimiter=',')
X = np.loadtxt('/home/brad/data/kth_test_hog.csv', delimiter=',')

#transpose to be (samples, features)
D = D.T
X = X.T

#copy
NNU_D = np.copy(D)
NNU_X = np.copy(X)

#python analysis
D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
D_mean = np.mean(D, axis=0)
D = D - D_mean

X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
X = X - D_mean

#true NNs
nbrs = np.argmax(np.dot(D, X.T), axis=0)

max_alpha = 25
max_beta = 50
storages = [Storage_Scheme.half, Storage_Scheme.mini, Storage_Scheme.micro,
            Storage_Scheme.two_mini]

assert False
verbose = True

for storage in storages:
    gen_colormap(max_alpha, max_beta, storage, NNU_X, NNU_D, nbrs,
                 verbose=verbose)

gen_colormap(max_alpha, max_beta, storage, NNU_X, NNU_D, nbrs,
             verbose=verbose, random=True)
