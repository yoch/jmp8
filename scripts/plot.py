import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from misc import *


if __name__ == '__main__':

    for key in ['en', 'de', 'it', 'es']:

        gold = load_keys('../keys/%s.test.gold.txt' % key)
        run1 = load_keys('../output/run1/%s.output.txt' % key)
        run2 = load_keys('../output/run2/%s.output.txt' % key)

        # "remove" missing words
        run1[run1==0.5] = float('nan')
        run2[run2==0.5] = float('nan')

        # scale the values
        run1 *= 4
        run2 *= 4

        # sort word pairs by similarity
        order = np.argsort(gold)

        fig = plt.figure()

        plt.ylim((-0.1,4.1))

        plt.plot(gold[order], '-', c='0.45', label='gold')
        plt.plot(run1[order], '+', c='0.4', ms=4, label='run 1')
        plt.plot(run2[order], 'x', c='0.15', ms=3, label='run 2')

        plt.ylabel('score')
        plt.xlabel('word pairs')

        plt.legend(loc='best')
        plt.title(key)

        plt.savefig('../misc/plot-%s.png' % key)
        plt.close()
