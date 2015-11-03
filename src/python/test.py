import numpy as np
import pscgen

nnu = pscgen.NNU(10, 10)
# nnu.build_index('/home/brad/data/notredame/small.csv')
# nnu.save('test.nnu')
nnu.load('test.nnu')

x = np.loadtxt('/home/brad/data/notredame/tiny.csv', delimiter=',')
print np.array(nnu.index(x))
