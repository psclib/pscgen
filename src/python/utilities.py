import numpy as np
import sys
from functools import partial

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import chi2_kernel
from numpy.lib.stride_tricks import as_strided as ast

def normalize(X):
    X = np.copy(X)
    norms = np.linalg.norm(X, axis=1)
    nonzero = np.where(norms != 0)
    X[nonzero] /= norms[nonzero][:, np.newaxis]

    return X


def bow_list(xs, N):
    xs_rep = []

    for item in xs:
        xs_rep.append(bow(item, minlength=N))

    xs_rep = np.array(xs_rep)

    return xs_rep


def bow(xs, minlength):
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

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.
     
    Parameters
        shape - an int, or a tuple of ints
     
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass


def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    Parameters:
    a - an n-dimensional numpy array
    ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
    of each dimension of the window
    ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
    amount to slide the window in each dimension. If not specified, it
    defaults to ws.
    flatten - if True, all slices are flattened, otherwise, there is an 
    extra dimension for each dimension of the input.

    Returns
    an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window. I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)
