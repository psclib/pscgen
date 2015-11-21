import glob

import numpy as np
from scikits.audiolab import wavread
from itertools import islice

def window(seq, n=2, s=1):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result


base_path = '/home/brad/data/voice/'
names = ['voice_surat', 'voice_lulu', 'voice_jie']

X, Y = [], []

for name in names:
    folder_path = base_path + name + '/'

    files = glob.glob(folder_path + '*.wav')

    for f in files:
        print f
        data, sample_frequency,encoding = wavread(f)
        data = np.array(list(window(data[:, 0], 100)))
        f = f.split('/')[-1].split('_')[1]
        X.append(data)
        Y.append(f)

# np.savez('/home/brad/data/surat_voice.npz', X=X, Y=Y)
