import sys
import json
import glob

import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.svm import SVC
from scikits.audiolab import wavread

from pscgen import NNU, name_to_storage
import pscgen
import utilities as util

def wav_to_np(folder_path, window_size=100, slide_size=12):
    if folder_path[-1] != '/':
        folder_path += '/'

    X, Y = [], []
    files = glob.glob(folder_path + '*.wav')
    for f in files:
        data, sample_frequency,encoding = wavread(f)
        data = np.array(list(util.sliding_window(data[:, 0], window_size,
                                                 slide_size)))
        f = f.split('/')[-1].split('_')[1]
        X.append(data)
        Y.append(f)

    return X, Y


def read_dataset(folder_path, dtype, window_size=100, slide_size=12):
    if dtype == 'wav':
        return wav_to_np(folder_path, window_size=window_size,
                         slide_size=slide_size)
    else:
        raise ValueError('{} is invalid file format'.format(dtype))


def parse_list(k, v, chunk_size):
    chunks = []
    dtype = type(v[0])
    v = str(v)
    k_size = len(v)
    k_pos = 1
    start_idx = 0

    #remove key size chars from chunk_size
    chunk_size -= (len(k) + len('type: ') + len('start: ') + len('end: ') + 40)

    while k_pos < k_size:
        next_comma = v[k_pos:k_pos+chunk_size].rfind(',')
        if next_comma == -1 or k_pos + chunk_size >= k_size:
            next_comma = chunk_size

        v_str = v[k_pos:k_pos+next_comma]
        if k_pos + next_comma + 1 > k_size:
            v_str = v_str[:-1]

        end_idx = start_idx + v_str.count(',') + 1
        chunk_d = {'type': k, 'start': start_idx, 'end': end_idx,
                   'buf': map(dtype, v_str.split(','))}
        chunks.append(chunk_d)
        k_pos += next_comma + 1
        start_idx = end_idx

    return chunks

def chunk_json(json_dict, chunk_size=None):
    '''
    Chunks json string into smaller sections. NOTE: custom -- not flexible
    '''
    if chunk_size == None:
        chunk_size = int(1e7)

    num_chunks = 0 
    chunk_dict = {'chunks':[]}
    for base_k in ['nnu', 'svm']:
        for k, v in json_dict[base_k].iteritems():
            if type(v) == list:
                lst_chunks = parse_list(k, v, chunk_size)
                chunk_dict['chunks'].extend(lst_chunks)
                num_chunks += len(lst_chunks)
            else:
                chunk_d = {'type': k, 'buf':v}
                chunk_dict['chunks'].append(chunk_d)
                num_chunks += 1
    chunk_dict['num_chunks'] = num_chunks

    return json.dumps(chunk_dict)


'''
Updater:
dtype (e.g. wav)
tr_folder_path (containing files of format classlabel_XXXXX.dtype)
D_atoms
alpha
beta
storage (e.g. 'mini' or 'half')
output_path
chunk_size (size in bytes, -1 for no chunks)
'''
args = json.loads(sys.argv[1])
storage = name_to_storage(args['storage'])
KMeans_tr_size = 200000
X, Y = read_dataset(args['tr_folder_path'], args['dtype'])


pipe = pscgen.Pipeline(100, 12)
pipe.fit(X, Y, args['D_atoms'], args['alpha'], args['beta'], storage)

assert False

X_Kmeans = np.vstack(X)[:KMeans_tr_size]

# Train D using KMeans
D = KMeans(n_clusters=args['D_atoms'], init_size=args['D_atoms']*3)
D.fit(X_Kmeans)
D = D.cluster_centers_
np.savetxt('/home/brad/data/voice_D_200.csv', D.T, delimiter=',', fmt='%2.6f')
D = util.normalize(D)
D_mean = np.mean(D, axis=0)
D = D - D_mean

#TODO: update to use D

svm_X = []
for x in X:
    x = util.normalize(x)
    x = x - D_mean
    nbrs = np.argmax(np.abs(np.dot(D, x.T)), axis=0)
    svm_X.append(util.bow(nbrs, args['D_atoms']))

clf = SVC(kernel='linear')
clf.fit(svm_X, Y)

assert False
svm_dict = {}
svm_dict['num_classes'] = len(set(Y))
svm_dict['num_clfs'] = len(clf.intercept_)
svm_dict['num_features'] = len(clf.coef_[0])
svm_dict['coefs'] = list(clf.coef_.flatten())
svm_dict['intercepts'] = list(clf.intercept_)

nnu = NNU(args['alpha'], args['beta'], storage)
nnu.build_index(D)

json_dict = {}
json_dict['nnu'] = nnu.to_dict()
json_dict['svm'] = svm_dict

if 'chunk_size' in args.keys():
    chunk_size = args['chunk_size']
else:
    chunk_size = None

json_str = chunk_json(json_dict, chunk_size=chunk_size)

with open(args['output_path'], 'w+') as fp:
    fp.write(json_str)
