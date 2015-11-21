import numpy as np
import glob

def in_train(sample_name):
    with open('kth_train_labels.txt', 'r') as fp:
        for line in fp:
            if sample_name in line:
                return 1

    with open('kth_test_labels.txt', 'r') as fp:
        for line in fp:
            if sample_name in line:
                return 0


    return -1

def get_class_label(sample_name):
    with open('kth_train_labels.txt', 'r') as fp:
        for line in fp:
            if sample_name in line:
                return int(line.split()[1])

    with open('kth_test_labels.txt', 'r') as fp:
        for line in fp:
            if sample_name in line:
                return int(line.split()[1])

    return -1


tr_files = glob.glob('kth/*.npy')
xs_tr = []
xs_t = []
ys_tr = []
ys_t = []

for f in tr_files:
    sample_name = f.split('/')[1]
    tr_or_t = in_train(sample_name)
    if tr_or_t == 1:
        xs_tr.append(np.load(f))
        ys_tr.append(get_class_label(sample_name))
    elif tr_or_t == 0:
        xs_t.append(np.load(f))
        ys_t.append(get_class_label(sample_name))

np.savez('kth_wang.npz', X_tr=xs_tr, X_t=xs_t, Y_tr=ys_tr, Y_t=ys_t)
