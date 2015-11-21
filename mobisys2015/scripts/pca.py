#PCA analysis for NNU

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator

import matplotlib
matplotlib.rcParams.update({'font.size': 22})

def intersect(*d):
    sets = iter(map(set, d))
    result = sets.next()
    for s in sets:
        result = result.intersection(s)
    return result

X = np.loadtxt('../data/kth_test_hog.csv', delimiter=',')
D = np.loadtxt('../data/D1500_hog.csv', delimiter=',')

X = X.T
D = D.T

D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
D_mean = np.mean(D, axis=0)
D = D - D_mean
X = X - D_mean
_, _, V = np.linalg.svd(D)
VD = np.dot(V, D.T)
VX = np.dot(V, X.T)

N = 5
linewidth = 2
x_idx = np.argmax(VX[0])
x_idx = np.random.permutation(len(X))[:100]

for i in range(N):
    label = 'span$(v_{' + str(i+1) + '})$'
    x = []
    for j in range(len(x_idx)):
        x.append(np.sort(np.abs(VX[i, x_idx[j]] - VD[i][:, np.newaxis]).flatten())[::-1])

    xmean = np.mean(x, axis=0)
    plt.plot(xmean, linewidth=linewidth, label=label)

plt.xlabel('Sorted Atoms by Distance')
plt.ylabel('Distance in Subspace Spanned by Principal Components')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.15/V_individual.png')
plt.clf()

for i in range(N):
    x = []
    for j in range(len(x_idx)):
        x.append(np.sort(np.sum(np.abs(VD[:i+1] - VX[:i+1, x_idx[j]][:, np.newaxis]), axis=0))[::-1])
    x = np.mean(x, axis=0)

    label = 'span$('
    for j in range(i):
        label += 'v_{' + str(j+1) + '},'
    label += 'v_{' + str(i+1) + '})$'
    plt.plot(x, linewidth=linewidth, label=label)

plt.xlabel('Sorted Atoms by Distance')
plt.ylabel('Distance in Subspace Spanned by Principal Components')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.15/V_combined.png')
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
spacing = 1
minorLocator = MultipleLocator(spacing)


for i in range(N):
    x = []
    for j in range(len(x_idx)):
        x.append(np.sort(np.sum(np.abs(VD[:i+1] - VX[:i+1, x_idx[j]][:, np.newaxis]), axis=0))[::-1])

    x = np.mean(x, axis=0)
    x = x[1480:]
    label = 'span$('
    for j in range(i):
        label += 'v_{' + str(j+1) + '},'
    label += 'v_{' + str(i+1) + '})$'
    ax1.plot(range(1480, 1500), x, linewidth=linewidth, label=label)

ax1.xaxis.set_minor_locator(minorLocator)
ax1.grid(b=True, which='minor', linewidth=1)
plt.xlabel('Sorted Atoms by Distance')
plt.ylabel('Distance in Subspace Spanned by Principal Components')
# plt.xlim([1480, 1500])
# plt.ylim([0.0, np.max(x)*1.1])
# plt.legend()
fig = plt.gcf()
fig.set_size_inches(18.5, 12.5)
plt.savefig('/home/brad/11.15/V_combined_zoom.png')
plt.clf()



N = 5

for i in range(N):
    label = '$v_{' + str(i+1) + '}^{T}X$'
    plt.plot(np.sort(VX[i, :])[::-1], linewidth=linewidth, label=label)

plt.xlabel('Sorted $v_i^{T}X$')
plt.ylabel('$v_i^{T}X$')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.15/sorted_vtx.png')
plt.clf()

for i in range(N):
    label = '$v_{' + str(i+1) + '}^{T}D$'
    plt.plot(np.sort(VD[i, :])[::-1], linewidth=linewidth, label=label)

plt.xlabel('Sorted $v_i^{T}D$')
plt.ylabel('$v_i^{T}D$')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('/home/brad/11.15/sorted_vtd.png')
plt.clf()


assert False


for i in range(N):
    x = np.sort(np.sum(np.abs(VD[:i+1] - VX[:i+1, x_idx][:, np.newaxis])[::-1], axis=0))
    plt.plot(x, label='Min VD0-' + str(i))

plt.legend()
plt.show()



N = 10
x_idx = np.argmax(VX[1])

for i in range(N):
    plt.plot(np.sort(np.abs(VX[i, x_idx] - VD[i])), label='V_' + str(i))


plt.legend()
plt.show()

for i in range(N):
    x = np.sort(np.sum(np.abs(VD[:i+1] - VX[:i+1, x_idx][:, np.newaxis]), axis=0))
    plt.plot(x, label='Max VD0-' + str(i))

plt.legend()
plt.show()



x_idx = np.argmin(VX[1])

for i in range(N):
    plt.plot(np.sort(np.abs(VX[i, x_idx] - VD[i])), label='V_' + str(i))


plt.legend()
plt.show()


for i in range(N):
    x = np.sort(np.sum(np.abs(VD[:i+1] - VX[:i+1, x_idx][:, np.newaxis]), axis=0))
    plt.plot(x, label='Min VD0-' + str(i))

plt.legend()
plt.show()




for i in range(1):
    x_idx = np.random.permutation(len(X))[0]

    # for i in range(N):
    #     plt.plot(np.sort(VD[i]), label='V_' + str(i) + 'D')

    # plt.legend()
    # plt.show()

    # for i in range(N):
    #     plt.plot(np.sort(VX[i]), label='V_' + str(i) + 'X')

    # plt.legend()
    # plt.show()

    for i in range(N):
        plt.plot(np.sort(np.abs(VX[i, x_idx] - VD[i])), label='V_' + str(i))


    plt.legend()
    plt.show()

    # for i in range(N):
    #     plt.plot(np.sort(np.abs(VX[i, x_idx] - VD[0])), label='VD0, X_' + str(i))

    # plt.legend()
    # plt.show()

    for i in range(N):
        x = np.sort(np.sum(np.abs(VD[:i+1] - VX[:i+1, x_idx][:, np.newaxis]), axis=0))
        plt.plot(x, label='VD0-' + str(i))

    plt.legend()
    plt.show()

    # plt.legend()
    # plt.show()

    # for i in range(N):
    #     x = np.sort(np.sum(np.abs(VD[:i+1] - VD[:i+1, dx][:, np.newaxis]), axis=0))
    #     plt.plot(x, label='VD0-' + str(i))

    # plt.legend()
    # plt.show()


assert False

N = 20
M = 1000
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
        sorted_nnu = np.argsort(np.abs(vx - VD[i]))
        # sorted_nnu = np.random.permutation(D.shape[1])
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
