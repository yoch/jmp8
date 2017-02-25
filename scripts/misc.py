#!/usr/bin/env python3

import numpy as np

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