import numpy as np
import matplotlib.pyplot as plt

from pscgen import NNU, Storage_Scheme, storage_stride

def gen_colormap(max_alpha, max_beta, gamma_pow, storage, iden, NNU_X, NNU_D,
                 nbrs, random=False, verbose=False):
    base_path = '/home/brad/11.15/'
    NNU_D = np.copy(NNU_D)

    if not random:
        s_stride = storage_stride(storage)
        max_alpha *= s_stride
        print max_alpha, max_beta, gamma_pow, storage
        nnu = NNU(max_alpha, max_beta, gamma_pow, storage)
        nnu.build_index(NNU_D)
    else:
        s_stride = 1

    count = np.zeros((max_alpha / s_stride, max_beta))

    print max_alpha
    alpha = 10
    beta = 30
    if not random:
        nnu_nbrs = nnu.index(NNU_X, alpha=alpha, beta=beta)
    else:
        nnu_nbrs = []
        for x in X:
            idxs = np.random.permutation(len(D))[:alpha*beta]
            nnu_nbrs.append(idxs[np.argmax(np.abs(np.dot(D[idxs], x)))])


    pct_found = len(np.where(nnu_nbrs == nbrs)[0]) / float(len(X))
    print pct_found
    return
    for alpha in range(s_stride, max_alpha+1, s_stride):
        for beta in range(1, max_beta+1):
            if not random:
                nnu_nbrs = nnu.index(NNU_X, alpha=alpha, beta=beta)
            else:
                nnu_nbrs = []
                for x in X:
                    idxs = np.random.permutation(len(D))[:alpha*beta]
                    nnu_nbrs.append(idxs[np.argmax(np.abs(np.dot(D[idxs], x)))])
            
            pct_found = len(np.where(nnu_nbrs == nbrs)[0]) / float(len(X))
            count[(alpha / s_stride) - 1, beta-1] = pct_found
            print alpha, beta, count[(alpha / s_stride)-1, beta-1]

    plt.imshow(count, interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(base_path + iden + 'color.png')
    plt.clf()
    np.save(base_path + iden + '_count', count)


D = np.loadtxt('/home/brad/data/D1500_hog.csv', delimiter=',')
X = np.loadtxt('/home/brad/data/kth_test_hog.csv', delimiter=',')

#transpose to be (samples, features)
D = D.T
X = X.T

# X = X[:100]

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

max_alpha = 5
max_beta = 5
storages = [Storage_Scheme.half, Storage_Scheme.mini, Storage_Scheme.micro,
            Storage_Scheme.two_mini, Storage_Scheme.four_micro, 0]
gammas = [16, 8, 4, 16, 16, 0]
names = ['half', 'mini', 'micro', 'two_mini', 'four_micro', 'random']

storages = [Storage_Scheme.two_mini]
gammas = [16]
names = ['two_mini']

verbose = True

for storage, gamma_pow, name in zip(storages, gammas, names):
    print name
    if name == 'random':
        gen_colormap(max_alpha, max_beta, gamma_pow, storage, name, NNU_X,
                     NNU_D, nbrs, verbose=verbose, random=True)
    else:
        gen_colormap(max_alpha, max_beta, gamma_pow, storage, name, NNU_X,
                     NNU_D, nbrs, verbose=verbose)
