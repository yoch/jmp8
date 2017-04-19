#!/usr/bin/env python3

from collections import defaultdict, Counter
from itertools import islice
from time import time
from contextlib import contextmanager
from scipy.sparse import csr_matrix
from datasets import load


MAXSAMPLES = 300000
MAXFEATURES = 100000
K = 4           # window size
SUFFIX = ''     # to distinguish between some variants



@contextmanager
def chrono(*args, **kw):
    print(*args, **kw)
    t1 = time()
    yield
    print(' ... elapsed %.3fs' % (time() - t1))

def ngrams(seq, n):
    return zip(*[islice(seq, k, None) for k in range(n)])

if __name__ == '__main__':
    import sys

    # language name
    key = sys.argv[1]

    with chrono('loading dataset', key):
        all_sents = [words for words in load(key)]
        print(len(all_sents), 'lines read')

    with chrono('count unique words'):
        uwords = Counter(filter(str.isalpha, (w for sent in all_sents for w in sent)))
        print(sum(uwords.values()), 'words', len(uwords), 'unique words')

    with chrono('get most used words'):
        retained_words = [w for w, _ in uwords.most_common(NWORDS)]
        retained = {w: i for i, w in enumerate(retained_words)}
        nwords = len(retained_words)
        print(nwords, 'words retained')

    NFEATURES = min(MAXFEATURES, len(retained_words))
    NSAMPLES = min(MAXSAMPLES, len(retained_words))

    with chrono('prepare matrix'):
        dct = defaultdict(float)
        for sent in all_sents:
            for x, *window in ngrams(sent, K):
                i = retained.get(x)
                if i is None:
                    continue
                for k, y in enumerate(window, 1):
                    j = retained.get(y)
                    if j is None:
                        continue
                    if i < NSAMPLES and j < NFEATURES:
                        dct[i,j] += 1 / k
                    if j < NSAMPLES and i < NFEATURES:
                        dct[j,i] += 1 / k


    with chrono('build CSR matrix'):
        I = np.fromiter((i for i,j in dct.keys()), dtype='i', count=len(dct))
        J = np.fromiter((j for i,j in dct.keys()), dtype='i', count=len(dct))
        data = np.fromiter(dct.values(), dtype='f', count=len(dct))
        M = csr_matrix((data, (I,J)), dtype='f')
        print(' matrix shape: ', M.shape, 'nnz:', M.nnz)

    skip = set()

    with chrono('save matrix'):
        fp = open(key + SUFFIX + '.svm', 'w', encoding='utf8')
        data, indices, indptr = M.data, M.indices, M.indptr
        for i in range(len(indptr)-1):
            start, stop = indptr[i:i+2]
            if start == stop:
                print('Warning: blank line at %d' % i)
                skip.add(i)
                continue
            line = ' '.join('%d:%g' % (indices[j], data[j])
                            for j in range(start, stop))
            fp.write(str(i))
            fp.write(' ')
            fp.write(line)
            fp.write('\n')
        fp.close()

    with chrono('save words'):
        fp = open(key + SUFFIX + '.words', 'w', encoding='utf8')
        for i, w in enumerate(islice(retained_words, NSAMPLES)):
            if i in skip:
                continue
            fp.write(w)
            fp.write('\n')
        fp.close()
