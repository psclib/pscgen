import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def intersect(*d):
    sets = iter(map(set, d))
    result = sets.next()
    for s in sets:
        result = result.intersection(s)
    return result

X = np.loadtxt('/home/brad/data/kth_test_hog.csv', delimiter=',')
D = np.loadtxt('/home/brad/data/D1500_hog.csv', delimiter=',')
D_mean = np.mean(D, axis=1)
D = (D.T - D_mean).T
X = (X.T - D_mean).T
_, _, V = np.linalg.svd(D.T)
VD = np.dot(V, D)
VX = np.dot(V, X)

for i in range(5):
    plt.plot(np.sort(VD[i])[::-1], label='V_' + str(i) + 'D')

plt.legend()
plt.show()

for i in range(5):
    plt.plot(np.sort(VX[i])[::-1], label='V_' + str(i) + 'X')

plt.legend()
plt.show()

assert False


N = 20
M = 30000
count = np.zeros((N, D.shape[1]))
cum_count = np.zeros((N, D.shape[1]))
cum_count_row = np.zeros((N, D.shape[1]))

#nested lists to see where each sample was found
found_samples = []
for i in range(N):
    found_samples.append([[] for i in range(D.shape[1])])

for i in range(N):
    for j, vx in enumerate(VX[i][:M]):
        max_idx = np.argmax(np.dot(X[:, j], D))
        # sorted_nnu = np.argsort(np.abs(vx - VD[i]))
        sorted_nnu = np.random.permutation(D.shape[1])
        found_idx = np.where(sorted_nnu == max_idx)[0]
        count[i, found_idx] += 1
        cum_count_row[i, found_idx:] += 1
        found_samples[i][found_idx].append(j)

for i in range(N):
    for j in range(D.shape[1]):
        l = [s[:j+1] for s in found_samples[:i+1]]
        # l = [[s for s in found_samples[i][:j+1]]]
        # l2 = [s[:20] for s in found_samples[:i]]
        # l = l + l2
        l = [item for sublist in l for item in sublist]        
        cum_count[i, j] = len(set.union(*map(set,l)))

cum_count = cum_count / float(M)
count_pct = (count.T / np.sum(count, axis=1)).T

for i in range(N):
    plt.plot(count[i], label=str('alpha: ' + str(i)))
    plt.legend()
    plt.ylabel('number of times nearest neighbor')
    plt.xlabel('beta table position')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('/home/brad/11.13.random/alpha' + str(i) + '.png')
    plt.clf()


plt.plot(np.sum(count, axis=0), label='combined over all tables')
plt.legend()
plt.ylabel('number of times nearest neighbor')
plt.xlabel('beta table position')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.13.random/overall_alpha.png')
plt.clf()

plt.imshow(count_pct[:, :200], interpolation='nearest')
plt.colorbar()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.13.random/nn_color.png')
plt.clf()

plt.imshow(count_pct[:, :200], interpolation='nearest', cmap=cm.Greys_r)
plt.colorbar()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.13.random/nn_grey.png')
plt.clf()

plt.imshow(cum_count[:, :200], interpolation='nearest')
plt.colorbar()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.13.random/cumulative_nn_color.png')
plt.clf()

plt.imshow(cum_count[:, :200], interpolation='nearest', cmap=cm.Greys_r)
plt.colorbar()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.13.random/cumulative_nn_grey.png')
plt.clf()
