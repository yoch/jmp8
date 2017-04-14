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

        fig, axes = plt.subplots(2, figsize=(8, 10), sharex=True, sharey=True)

        plt.ylabel('score')
        plt.xlabel('word pairs')
        
        plt.ylim((-0.1,4.1))
        #plt.title(key)

        axes[0].plot(gold[order], '-', c='0.1', label='gold')
        axes[0].plot(run1[order], '+', c='0.3', ms=5.5, label='version 1')
        axes[0].legend(loc='best')

        axes[1].plot(gold[order], '-', c='0.1', label='gold')
        axes[1].plot(run2[order], 'x', c='0.2', ms=4.5, label='version 2')
        axes[1].legend(loc='upper left')

        #fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.15)
        plt.savefig('../misc/plot-%s.pdf' % key, bbox_inches="tight")
        plt.close()
