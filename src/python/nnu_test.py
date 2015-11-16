import numpy as np
import matplotlib.pyplot as plt

from pscgen import NNU, Storage_Scheme, storage_stride

def idx3d(i, j, k, rows, cols):
    return i * rows * cols + j * rows + k

def gen_colormap(max_alpha, max_beta, gamma_pow, storage, iden, NNU_X, NNU_D,
                 nbrs, random=False, verbose=False):
    base_path = '/home/brad/11.15/'
    NNU_D = np.copy(NNU_D)

    if not random:
        s_stride = storage_stride(storage)
        nnu = NNU(max_alpha, max_beta, gamma_pow, storage)
        nnu.build_index(NNU_D)
    else:
        s_stride = 1

    count = np.zeros((max_alpha / s_stride, max_beta))

    alpha = 2
    beta = 2
    if not random:
        nnu_nbrs, ABs, runtime = nnu.index(NNU_X, alpha=alpha, beta=beta, detail=True)
    else:
        nnu_nbrs = []
        for x in X:
            idxs = np.random.permutation(len(D))[:alpha*beta]
            nnu_nbrs.append(idxs[np.argmax(np.abs(np.dot(D[idxs], x)))])


    pct_found = len(np.where(nnu_nbrs == nbrs)[0]) / float(len(X))
    print pct_found, ABs, runtime
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


# idxs = []
# for i in range(256):
#     for j in range(10):
#         for k in range(30):
#             idxs.append(idx3d(j, i, k, 30, 256))

# assert False
D = np.loadtxt('/home/brad/data/D1500_hog.csv', delimiter=',')
X = np.loadtxt('/home/brad/data/kth_test_hog.csv', delimiter=',')

#transpose to be (samples, features)
D = D.T
X = X.T

X = X[:300]

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

max_alpha = 15
max_beta = 30
storages = [Storage_Scheme.half, Storage_Scheme.mini, Storage_Scheme.micro,
            Storage_Scheme.two_mini, Storage_Scheme.four_micro]


nnu = NNU(max_alpha, max_beta, Storage_Scheme.four_micro)
nnu.build_index(NNU_D)
nnu_nbrs = nnu.index(NNU_X)
print len(np.where(nbrs == nnu.index(NNU_X))[0])

assert False
verbose = True

for storage, gamma_pow, name in zip(storages, gammas, names):
    print name
    if name == 'random':
        gen_colormap(max_alpha, max_beta, gamma_pow, storage, name, NNU_X,
                     NNU_D, nbrs, verbose=verbose, random=True)
    else:
        gen_colormap(max_alpha, max_beta, gamma_pow, storage, name, NNU_X,
                     NNU_D, nbrs, verbose=verbose)
