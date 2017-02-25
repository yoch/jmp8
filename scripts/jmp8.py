#!/usr/bin/env python3

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.utils.extmath import randomized_svd
from svmloader import *
from misc import *


USE_SVD = False

if not USE_SVD:
    from scipy.sparse import vstack
else:
    vstack = np.vstack


# helpers functions with CSR matrix
def div_cols(m, v):
    j_indices = m.indices
    m.data /= v[j_indices]

def div_rows(m, v):
    i_indices = np.repeat(np.arange(0, len(m.indptr)-1), np.diff(m.indptr))
    m.data /= v[i_indices]

def ppmi(m):
    #m = m.copy()

    sj = m.sum(0).A1
    si = m.sum(1).A1

    m.data *= m.nnz
    div_cols(m, sj)
    div_rows(m, si)

    m.data = np.log(m.data)

    m.data[m.data<0] = 0
    m.eliminate_zeros()

    return m


def get_vec(w, inv, M):
    # if the word is known
    if w in inv:
        return M[inv[w]]
    # split the expression
    ws = w.split(' ')
    if len(ws) > 1:
        f = [elt in inv for elt in ws]
        # if all words are known
        if all(f):
            # return the sum of vectors
            return sum(M[inv[e]] for e in ws)
        else:
            print('"%s" (%s) not found' % (w, ','.join([wd for wd, b in zip(ws, f) if not b])), file=sys.stderr)
            return
    print('"%s" not found' % w, file=sys.stderr)


def evaluate(wpairs, inv, M):
    A = np.full(shape=len(wpairs), fill_value=0.5)
    l = []  # list of vectors pairs
    f = []  # filter for known pairs
    for w1, w2 in wpairs:
        u = get_vec(w1, inv, M)
        v = get_vec(w2, inv, M)
        if u is None or v is None:
            f.append(False)
        else:
            f.append(True)
            l.append((u, v))
    print(f.count(False), 'missing results (use 0.5)', file=sys.stderr)
    I, J = map(vstack, zip(*l))
    dst = paired_cosine_distances(I, J)
    A[np.array(f)] = 1 - dst
    return A


if __name__ == '__main__':
    import sys

    assert len(sys.argv) >= 4, 'bad arguments'
    fmatrix = sys.argv[1]
    fwords = sys.argv[2]
    fdata = sys.argv[3]
    #fkeys = sys.argv[4]

    # load the raw matrix
    X = load_matrix(fmatrix)
    print(X.shape, X.nnz, file=sys.stderr)

    # compute its PPMI
    X = ppmi(X)
    print(X.shape, X.nnz, file=sys.stderr)

    if USE_SVD:
        X, _, _ = randomized_svd(X, n_components=500, n_iter=4, random_state=None)
        print(X.shape, file=sys.stderr)

    # load words
    y = load_words(fwords)
    # make word to index dict
    inv = {w: i for i, w in enumerate(y)}

    # load test set
    wpairs = load_data(fdata)

    results = evaluate(wpairs, inv, X)
    for val in results:
        print(val)

    #keys = load_keys(fkeys)

    #print('Pearson:', pearsonr(keys, results)[0])
    #print('Spearman:', spearmanr(keys, results)[0])
