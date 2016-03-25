import pickle

import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

import utilities as util
from pscgen import NNUForest, NNUDictDim, normalize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'axes.labelsize': 12,
   'text.fontsize': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [10, 10]})


def bow(xs, minlength):
    counts = np.bincount(xs, minlength=minlength)
    return counts/np.linalg.norm(counts)



class BoostedNN(object):
    def __init__(self, n_atoms, n_nodes=5, n_features=10):
        self.n_atoms = n_atoms
        self.n_nodes = n_nodes
        self.n_features = n_features


    def fit(self, X, Y):
        self.class_labels = np.array(sorted(set(Y)))
        D = KMeans(n_clusters=self.n_atoms, init_size=self.n_atoms*3)
        D.fit(np.vstack(X)[:200000])
        D = D.cluster_centers_
        D = normalize(D)
        self.D_mean = np.mean(D, axis=0)
        self.D = D - self.D_mean
        self.D_idxs = []
        self.clfs = []

        for i in xrange(self.n_nodes):
            idxs = np.random.permutation(self.n_atoms)[:self.n_features]
            idxs = np.random.permutation(len(X[0][0]))[:self.n_features]
            enc_X = []
            for x in X:
                x = normalize(x)
                x = x - self.D_mean
                enc_X.append(bow(np.argmax(np.abs(np.dot(self.D[:, idxs], x.T[idxs, :])), axis=0), self.n_atoms))


            clf = LinearSVC()
            clf.fit(enc_X, Y)
            self.clfs.append(clf)
            self.D_idxs.append(idxs)

    def predict(self, X):
        predictions = []
        for idxs, clf in zip(self.D_idxs, self.clfs):
            enc_X = []
            for x in X:
                x = normalize(x)
                x = x - self.D_mean
                enc_X.append(bow(np.argmax(np.abs(np.dot(self.D[:, idxs], x.T[idxs, :])), axis=0), self.n_atoms))

            predictions.append(clf.decision_function(enc_X))

        predictions = np.array(predictions)

        return self.class_labels[np.argmax(np.sum(predictions, axis=0), axis=1)]




linewidth = 2
fig_name = 'boosted_dim'
KMeans_tr_size = 200000
D_atoms = 500
ws = 50
subsample_pcts = [1., 1., 1., 0.5]

X_flat, Y = util.wav_to_np('/home/brad/data/robot/')
X = []

num_folds = 10
sss = StratifiedShuffleSplit(Y, num_folds, test_size=0.7, random_state=0)

for x in X_flat:
    X.append(util.sliding_window(x, ws, 5))

plt_dict = {}
features = [2, 5, 10, 20]
nodes = [1, 2, 5, 10, 15, 20]

for f in features:
    plt_dict[f] = []
    for n in nodes:
        print f, n
        accs = []
        for train_index, test_index in sss:
            X_train = [X[i] for i in train_index]
            X_test = [X[i] for i in test_index]
            bnn = BoostedNN(D_atoms, n, f)
            bnn.fit(X_train, Y[train_index])
            Y_pred = bnn.predict(X_test)
            acc = accuracy_score(Y[test_index], Y_pred)
            print acc
            accs.append(acc)

        plt_dict[f].append(np.mean(accs))


fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for f in features:
    ax.plot(nodes, plt_dict[f], '-o', linewidth=linewidth, label=str(f))

plt.xlabel('Number of Classifier Nodes')
plt.ylabel('Classification Accuracy')
ax.set_ylim((0, 1.0))
plt.legend(loc='lower right')
fig = plt.gcf()
plt.savefig('../new/' + fig_name + '.pdf')
plt.clf()
