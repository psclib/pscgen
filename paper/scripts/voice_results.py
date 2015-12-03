import sys
import json
import glob

from scikits.audiolab import wavread

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import numpy as np
import utilities as util

import pscgen

def float_audio_to_int16(f):
    f = f * 32768 
    if f > 32767:
        f = 32767
    if f < -32768:
        f = -32768

    return int(f)


def int16_audio_to_float(i):
    f = float(i) / float(32768)
    if f > 1:
        f = 1
    if f < -1:
        f = -1

    return f


def wav_to_np(folder_path, window_size=100, slide_size=12):
    if folder_path[-1] != '/':
        folder_path += '/'

    X, Y, X_normal = [], [], []
    files = glob.glob(folder_path + '*.wav')
    for f in files:
        data, sample_frequency,encoding = wavread(f)
        data = data[:, 0]
        X_normal.append(data)
        data = np.array(list(util.sliding_window(data, window_size,
                                                 slide_size)))
        f = f.split('/')[-1].split('_')[1]
        X.append(data)
        Y.append(f)

    return np.array(X), np.array(Y), np.array(X_normal)

args = json.loads(sys.argv[1])
storage = pscgen.name_to_storage(args['storage'])
X, Y, X_flat = wav_to_np(args['tr_folder_path'])
num_folds = 3
acc = 0.0

sss = StratifiedShuffleSplit(Y, num_folds, test_size=0.5, random_state=0)
alphas = [1, 2, 3, 4, 5, 5, 5, 10, 10, 15]
betas = [1, 1, 1, 1, 1, 2, 4, 5, 10, 10]
alphas = [1, 2, 3, 4, 5, 5, 5, 10, 10]
betas = [1, 1, 1, 1, 1, 2, 4, 5, 10]
Ns = [1, 2, 3, 4, 5, 10, 20, 50, 100, 150]
for alpha, beta in zip(alphas, betas):
    for train_index, test_index in sss:
        pipe = pscgen.Pipeline(100, 12)
        pipe.fit(X_flat[train_index], Y[train_index], args['D_atoms'],
                 alpha, beta, storage)

        Y_pred = [pipe.classify(X_flat[idx]) for idx in test_index]
        acc += accuracy_score(Y[test_index], Y_pred)

    acc = acc / float(num_folds)
    print alpha, beta, acc
    acc = 0.0
