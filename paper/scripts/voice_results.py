import sys
import json

from itertools import product
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import numpy as np
import utilities as util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'axes.labelsize': 12,
   'text.fontsize': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [4.5, 4.5]})

import pscgen

def num_ops(N, M):
    return N * (2 * M - 1)

args = json.loads(sys.argv[1])
storage = pscgen.name_to_storage(args['storage'])
X, Y, X_flat = util.wav_to_np(args['tr_folder_path'], window_size=50)
num_folds = 15
acc = 0.0
max_atoms = 1000

sss = StratifiedShuffleSplit(Y, num_folds, test_size=0.7, random_state=0)
alphas = [1, 2, 3, 4, 5, 5, 5, 5, 5, 10, 10, 10, 15, 20, 20, 20, 20, 25]
betas = [1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 7, 10, 10, 10, 12, 15, 20, 20]
Ns = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 70, 100, 150, 200, 240, 300, 400, 500]

alphas = [1, 2, 3, 4, 5, 5, 5, 5, 5, 10, 10, 10, 15, 20, 20]
betas = [1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 7, 10, 10, 10, 12]
Ns = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 70, 100, 150, 200, 240]

accs = {}
Ds = [500, 750, 1000]
# Ds = [750]
N_D = 750
ws = 30
ss = 20

# for ws, ss in product([10, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100, 200], [20]):
# # for N_D in  Ds:
#     for train_index, test_index in sss:
#         pipe = pscgen.Pipeline(ws, ss, max_classes=20)
#         pipe.fit(X_flat[train_index], Y[train_index], N_D, max_atoms,
#                  20, 12, storage, enc_type='nnu_pca')

#         Y_pred = [pipe.classify(X_flat[idx]) for idx in test_index]
#         acc += accuracy_score(Y[test_index], Y_pred)

#     acc = acc / float(num_folds)
#     print ws, ss, acc
#     acc = 0.0
# assert False


for enc_t in ['nnu', 'nnu_pca']:
    accs[enc_t] = []
    for alpha, beta in zip(alphas, betas):
        for train_index, test_index in sss:
            pipe = pscgen.Pipeline(ws, ss, max_classes=20)
            pipe.fit(X_flat[train_index], Y[train_index], N_D, max_atoms,
                     alpha, beta, storage, enc_type=enc_t)

            Y_pred = [pipe.classify(X_flat[idx]) for idx in test_index]
            acc += accuracy_score(Y[test_index], Y_pred)



        acc = acc / float(num_folds)
        print alpha, beta, acc
        accs[enc_t].append(acc)
        acc = 0.0



enc_t = 'nns'
accs[enc_t] = []
for D in Ns:
    for train_index, test_index in sss:
        pipe = pscgen.Pipeline(ws, ss, max_classes=20)
        pipe.fit(X_flat[train_index], Y[train_index], D, max_atoms,
                 1, 1, storage, enc_type=enc_t)

        Y_pred = [pipe.classify(X_flat[idx]) for idx in test_index]
        acc += accuracy_score(Y[test_index], Y_pred)

    acc = acc / float(num_folds)
    print  acc
    accs[enc_t].append(acc)
    acc = 0.0



ws = 40
names = {'nnu': 'NNU', 'nnu_pca': 'NNU-PCA-DR', 'nns': 'NNS'}
styles = {'nnu': '--', 'nnu_pca': '-', 'nns': '-.'}
colors = {'nnu': 'g', 'nnu_pca': 'b', 'nns': 'r'}
linewidth = 2

fig_name = 'voice_embed'
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
max_x = num_ops(Ns[-1], ws) / 1000.

for enc_t in ['nns', 'nnu', 'nnu_pca']:
    num_operations = []
    if enc_t == 'nnu':
        for alpha, beta in zip(alphas, betas):
            num_operations.append(num_ops(alpha*beta, ws) + 2*ws - 1)
    elif enc_t == 'nns':
        for N in Ns:
            num_operations.append(num_ops(N, ws))
    elif enc_t == 'nnu_pca':
        for alpha, beta in zip(alphas, betas):
            num_operations.append(num_ops(alpha*beta, alpha) + 2*ws - 1)

    num_operations = [b/1000. for b in num_operations]
    print num_operations

    ax.plot(num_operations, accs[enc_t], color=colors[enc_t],
            linestyle=styles[enc_t], label=names[enc_t], linewidth=linewidth)

ax.plot(np.linspace(0, max_x, 10000), [0.90]*10000, color='k',
        linestyle=':', linewidth=1.5)

# ax.set_xscale('log')
plt.xlabel('Number of Arithmetic Operations (Kops)')
plt.ylabel('Classification Accuracy')
plt.xlim((0, max_x))
ax.set_ylim(ymin=0)
plt.legend(loc='lower right')
fig = plt.gcf()
plt.savefig('../figures/' + fig_name + '_accuracy.pdf')
plt.clf()


fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
gamma = 256
precision = 4

for enc_t in ['nnu', 'nnu_pca']:
    num_bytes = []
    if enc_t == 'nnu':
        for alpha, beta in zip(alphas, betas):
            # num_bytes.append((N_D*ws*precision) + alpha*beta*gamma*2)
            num_bytes.append(alpha*beta*gamma*2)
    elif enc_t == 'nns':
        for N in Ns:
            num_bytes.append(N*ws*precision)
    elif enc_t == 'nnu_pca':
        for alpha, beta in zip(alphas, betas):
            # num_bytes.append((N_D*alpha*precision) + alpha*beta*gamma*2)
            num_bytes.append(alpha*beta*gamma*2)

    num_bytes = [b/1024. for b in num_bytes]
    ax.plot(num_bytes, accs[enc_t], color=colors[enc_t],
            linestyle=styles[enc_t], label=names[enc_t], linewidth=linewidth)

ax.plot(np.linspace(0, 250, 10000), [0.90]*10000, color='k',
        linestyle=':', linewidth=1.5)

# ax.set_xscale('log')
plt.xlabel('NNU Table Size (KB)')
plt.ylabel('Classification Accuracy')
ax.set_ylim(ymin=0)
plt.legend(loc='lower right')
fig = plt.gcf()
plt.savefig('../figures/' + fig_name + '_memory.pdf')
plt.clf()


nns_runtimes = [55331, 69088, 80149, 92409, 103556, 161751, 228104,
                280335, 353295, 640379, 894908, 1281273, 1968535, 2665573,
                3198880]

nnu_runtimes = [73571, 97133, 123142, 148929, 175670, 241990, 303221, 374466,
                421609, 790751, 1001498, 1299092, 1697016, 2097595, 2325473]
nnu_pca_runtimes = [54453, 65810, 78039, 93899, 105207, 117743, 122530, 130437,
                    139432, 287878, 331896, 404186, 654933, 970055, 1054998]

device_rt = {'nns': nns_runtimes, 'nnu': nnu_runtimes,
             'nnu_pca': nnu_pca_runtimes}
max_x = device_rt['nns'][-1] / 1000000.
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 

for enc_t in ['nns', 'nnu', 'nnu_pca']:
    rt = [r/1000000. for r in device_rt[enc_t]]
    ax.plot(rt, accs[enc_t], color=colors[enc_t],
            linestyle=styles[enc_t], label=names[enc_t], linewidth=linewidth)

ax.plot(np.linspace(0, max_x, 10000), [0.90]*10000, color='k',
        linestyle=':', linewidth=1.5)

# ax.set_xscale('log')
plt.xlabel('Average Runtime per Query (seconds)')
plt.ylabel('Classification Accuracy')
plt.xlim((0, max_x))
# plt.grid()
ax.set_ylim(ymin=0)
plt.legend(loc='lower right')
fig = plt.gcf()
plt.savefig('../figures/' + fig_name + '_runtime.pdf')
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 

for enc_t in ['nns', 'nnu', 'nnu_pca']:
    num_operations = []
    if enc_t == 'nnu':
        for alpha, beta in zip(alphas, betas):
            num_operations.append(num_ops(alpha*beta, ws) + 2*ws - 1)
    elif enc_t == 'nns':
        for N in Ns:
            num_operations.append(num_ops(N, ws))
    elif enc_t == 'nnu_pca':
        for alpha, beta in zip(alphas, betas):
            num_operations.append(num_ops(alpha*beta, alpha) + 2*ws - 1)

    rt = [r/1000000. for r in device_rt[enc_t]]
    num_operations = [b/1000. for b in num_operations]

    ax.plot(rt, num_operations, color=colors[enc_t],
            linestyle=styles[enc_t], label=names[enc_t], linewidth=linewidth)

plt.xlabel('Average Runtime per Query (seconds)')
plt.ylabel('Number of Arithmetic Operations (Kops)')
plt.xlim((0, max_x))
# plt.grid()
ax.set_ylim(ymin=0)
plt.legend(loc='lower right')
fig = plt.gcf()
plt.savefig('../figures/' + fig_name + '_ops_rt.pdf')
plt.clf()
