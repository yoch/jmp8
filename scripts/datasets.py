import re
from sys import intern


def load(key, maxi=None):
    rxp = re.compile(r'\[\[\d+\]\]')

    with open('%s/full.txt' % key, 'r', encoding='utf8') as fp:
        for i, line in enumerate(fp):
            if maxi is not None and i > maxi:
                break
            line = line.strip()
            if not line or rxp.match(line):
                continue
            words = [intern(w) for w in line.split()]
            yield words
