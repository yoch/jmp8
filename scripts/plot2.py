from collections import Counter, OrderedDict
from itertools import product, combinations_with_replacement
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from misc import *


plt.rcParams['image.cmap'] = 'gray'


def get_nwords(wpairs):
    lst = []
    for pair in wpairs:
        w1, w2 = map(lambda s: s.split(' '), pair)
        lst.append([len(w1), len(w2)])
    return np.array(lst)

if __name__ == '__main__':
    import sys

    run = 2 # sys.argv[1]

    dct = OrderedDict()

    for key in ['en', 'de', 'it', 'es']:

        data = load_data('../data/%s.test.data.txt' % key)
        gold = load_keys('../keys/%s.test.gold.txt' % key)
        out  = load_keys('../output/run%s/%s.output.txt' % (run, key))

        nwords = get_nwords(data)
        #nwords_max = nwords.max(axis=1)

        l = []
        for ij in combinations_with_replacement([1,2,3], r=2):
            i, j = ij
            f = ((nwords == (i,j)).all(axis=1) | (nwords == (j,i)).all(axis=1)) & (out != 0.5)
            n = np.count_nonzero(f)
            if n <= 2:
                continue
            p = pearsonr(gold[f], out[f])[0]
            s = spearmanr(gold[f], out[f])[0]
            #print(' Pearson:', p, ' Spearman:', s)

            l.append((ij, n, p, s))

        dct[key] = l

    f, axarr = plt.subplots(4, 2, figsize=(10, 9), sharex=True, sharey=True)
    axarr[0,0].set_title('Pearson')
    axarr[0,1].set_title('Spearman')

    for i, (label, lst) in enumerate(dct.items()):
        axarr[i,0].set_ylabel(label, labelpad=20, rotation=0, size='large')

        xy, nb, pearson, spearman = zip(*lst)
        #print(label, nb, pearson, spearman)
        x, y = zip(*xy)
        axarr[i,0].scatter(x, y, nb, c=pearson, label=label, vmin=0.5, vmax=0.8)
        s = axarr[i,1].scatter(x, y, nb, c=spearman, label=label, vmin=0.5, vmax=0.8)
        
        for j in range(2):
            axarr[i,j].set_xticks([1,2,3])
            axarr[i,j].set_yticks([1,2,3])

    f.colorbar(s, ax=axarr.ravel().tolist())

    #plt.legend()

    plt.savefig('../misc/multiwords-comparison-run%s.png' % run, bbox_inches="tight")
