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

X = X.T
D = D.T

D = D / np.linalg.norm(D, axis=1)[:, np.newaxis]
X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
D_mean = np.mean(D, axis=0)
D = D - D_mean
X = X - D_mean
_, _, V = np.linalg.svd(D)
# V = np.random.random(V.shape)
# V = V / np.linalg.norm(V, axis=1)[:, np.newaxis]
VD = np.dot(V, D.T)
VX = np.dot(V, X.T)

assert False
N = 10
x_idx = np.argmax(VX[0])

for i in range(N):
    plt.plot(np.sort(np.abs(VX[i, x_idx] - VD[i])), label='V_' + str(i))


plt.legend()
plt.show()

for i in range(N):
    x = np.sort(np.sum(np.abs(VD[:i+1] - VX[:i+1, x_idx][:, np.newaxis]), axis=0))
    plt.plot(x, label='Max VD0-' + str(i))

plt.legend()
plt.show()



x_idx = np.argmin(VX[0])

for i in range(N):
    plt.plot(np.sort(np.abs(VX[i, x_idx] - VD[i])), label='V_' + str(i))


plt.legend()
plt.show()


for i in range(N):
    x = np.sort(np.sum(np.abs(VD[:i+1] - VX[:i+1, x_idx][:, np.newaxis]), axis=0))
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
