import numpy as np
import sys
from functools import partial

import spams
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import chi2_kernel

# import nnu

def train_dict(data, components, sparsity, verbose=False):
    D = spams.trainDL(np.asfortranarray(data.T),
                      K=components,
                      lambda1=sparsity,
                      mode=3,
                      verbose=verbose,
                      iter=50,
                      posD=False)
    return np.asfortranarray(D)


def encode(data, D, k):
    return np.array(spams.omp(np.asfortranarray(data.T), D, L=k).todense())

def compute_bow(xs, N):
    xs_rep = []

    for item in xs:
        xs_rep.append(bag_rep(item, minlength=N))

    xs_rep = np.array(xs_rep)

    return xs_rep


def bag_rep(xs, minlength):
    counts = np.bincount(xs, minlength=minlength)
    return counts/np.linalg.norm(counts)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

    return i + 1

def predict_linear(tr_x, tr_y, t_x, t_y, batch=False, verbose=False):
    max_acc = 0
    tr_acc = 0
    cm = None
    tr_cm = None
    tuned_c = None

    #screen C
    # for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
    for C in [0.1]:
        if verbose:
            print 'Training with C: ', C

        clf = SVC(kernel='linear', C=C)
        clf.fit(tr_x, tr_y)
        y_pred = clf.predict(t_x)
        new_acc = accuracy_score(t_y, y_pred)
        if verbose:
            print 'Acc: ', new_acc

        if new_acc > max_acc:
            tr_y_pred = clf.predict(tr_x)
            tr_acc = accuracy_score(tr_y, tr_y_pred)
            tr_cm = confusion_matrix(tr_y, tr_y_pred)
            max_acc = new_acc
            cm = confusion_matrix(t_y, y_pred)
            tuned_c = C

    if verbose:
        print 'C: ', tuned_c
        print 'Training'
        print 'Accuracy:', tr_acc
        print tr_cm
        print ''

        print 'Testing'
        print 'Accuracy:', max_acc
        print cm
    else:
        sys.stdout.write(str('%2.3f' % max_acc))


def predict_chi2(tr_x, tr_y, t_x, t_y, batch=False, verbose=False):
    max_acc = 0
    tr_acc = 0
    cm = None
    tr_cm = None
    tuned_c, tuned_gamma = None, None

    #screen of gamma and C
    for g in [0.01, 0.05, .1]:
        for C in [1, 10, 100]:
            clf = SVC(kernel=partial(chi2_kernel, gamma=g), C=C)
            clf.fit(tr_x, tr_y)
            y_pred = clf.predict(t_x)
            new_acc = accuracy_score(t_y, y_pred)

            if new_acc > max_acc:
                tr_y_pred = clf.predict(tr_x)
                tr_acc = accuracy_score(tr_y, tr_y_pred)
                tr_cm = confusion_matrix(tr_y, tr_y_pred)
                max_acc = new_acc
                cm = confusion_matrix(t_y, y_pred)
                tuned_c = C
                tuned_gamma = g

    if verbose:
        print 'gamma, C: ', tuned_gamma, tuned_c
        print 'Training'
        print 'Accuracy:', tr_acc
        print tr_cm
        print ''

        print 'Testing'
        print 'Accuracy:', max_acc
        print cm

    return max_acc
