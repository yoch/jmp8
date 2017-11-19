#!/usr/bin/env python3

import re
import numpy as np
from svmloader import *
from time import time
from contextlib import contextmanager


@contextmanager
def chrono(*args, **kw):
    print(*args, **kw)
    t1 = time()
    yield
    print(' ... elapsed %.3fs' % (time() - t1))


def load_data(filename):
    pairs = []
    with open(filename, 'r', encoding='utf8') as fp:
        for line in fp:
            w1, w2 = line.strip().split('\t')
            pairs.append((w1, w2))
    return pairs

def load_keys(filename):
    with open(filename, 'r', encoding='utf8') as fp:
        keys = [float(line.strip()) for line in fp]
    return np.array(keys)

def load_words(filename):
    with open(filename, 'r', encoding='utf8') as fp:
        return [line.strip() for line in fp]

def load_matrix(filename):
    X, y = load_svmfile(filename, dtype='f')
    return X

def load_questions(filename):
    with open(filename, 'r', encoding='utf8') as fp:
        return [line.strip().split() for line in fp 
                if not line.startswith(':')]

def load_synonyms(filename, k=4):
    with open(filename, 'r', encoding='utf8') as fp:
        return [[w.strip() for w in line.strip().split('|')]
                for line in fp
                    if not line.startswith('#') 
                    and line.count('|') == k]

def get_syn_in_contexts(filename):
    with open(filename) as fp:
        lst = [line.strip(' ",\n').split('|', 1) for line in fp]
        contexts, candidates = zip(*lst)
        questions = re.findall(r'\[(.*)\]', '\n'.join(contexts))
        Q = [[q] + [w.strip() for w in cand.split('|')]
             for q, cand in zip(questions, candidates)]
        return Q, contexts

def get_kmax(A, k):
    # bidimensional k-argmax on last axis
    rows,_ = np.indices((A.shape[0],k))
    Ap = np.argpartition(A, -k)[:,-k:]
    return Ap[rows,np.argsort(A[rows,Ap])[:,::-1]]

def get_kmin(A, k):
    # bidimensional k-argmin on last axis
    rows,_ = np.indices((A.shape[0],k))
    Ap = np.argpartition(A, k)[:,:k]
    return Ap[rows,np.argsort(A[rows,Ap])]


def pmi(m, flog=np.log, copy=True):
    "Calculate Pointwise Multiple Information of a sparse CSR matrix."

    def div_cols(v):
        j_indices = m.indices
        m.data /= v[j_indices]

    def div_rows(v):
        i_indices = np.repeat(np.arange(0, len(m.indptr)-1), np.diff(m.indptr))
        m.data /= v[i_indices]

    if copy:
        m = m.copy()

    sj = m.sum(0).A1
    si = m.sum(1).A1

    m.data *= m.sum()
    div_cols(sj)    # m = m.multiply(1 / sj)
    div_rows(si)

    flog(m.data, m.data)

    return m

def ppmi(m, threshold=0., **kw):
    "Compute Positive PMI"

    assert threshold >= 0

    m = pmi(m, **kw)

    m.data[m.data<=threshold] = 0
    m.eliminate_zeros()

    return m
