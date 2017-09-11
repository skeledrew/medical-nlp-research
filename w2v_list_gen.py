#! /home/aphillips5/envs/nlpenv/bin/python3


import sys, pdb
from os.path import exists

from pexpect.replwrap import REPLWrapper

from common import *


class Distance(): #Distance(REPLWrapper):

    def __init__(self, cmd='', prompt=None, rx=None):
        if not cmd: cmd = '{} {}'.format(dist, mimic_bin)
        if not prompt: prompt = 'Enter word or sentence (EXIT to break): '

        if isinstance(cmd, str):
            self._repl = REPLWrapper(cmd, prompt, None)

        else:
            self._repl = cmd
        #super(Distance, self).__init__(cmd, prompt, None, prompt)
        self.nodes = []
        self.rx = rx if rx else '\s+(?P<Word>[\w-]+)\t+.*'

    def find(self, words, num, max_levels=1, unique=True, level=0):
        first = self._repl.run_command(words, None).split('\n')[5:num+5]
        first = [re.match(self.rx, line).group('Word') for line in first]

        if level < max_levels:
            self.nodes = [Distance(self._repl) for _ in range(len(first))]
            rest = [self.nodes[idx].find(first[idx], num, max_levels, level=level+1) for idx in range(len(first))]
            first.extend(rest)
            first = [word for sublist in first if isinstance(sublist, list) for word in sublist]
            #first = [word for word in first if len(word) > 1]
            #if unique: first = list(set(first))
        if level == 0: print(first, len(first))
        return first

def list_gen(seed, number, levels):
    words = [seed]
    curr_level = 0
    distance = Distance()
    number = int(number)
    levels = int(levels)

    while words:
        distance.find(words.pop(0), number, levels)
    #distance.finish()

mimic_bin = '/NLPShare/Lib/Word2Vec/Models/mimic.bin'
dist = '/NLPShare/Lib/Word2Vec/word2vec/distance'

if __name__ == '__main__':
    args = sys.argv[1:]

    try:
        if len(args) >= 3: list_gen(args[0], args[1], args[2])
        commit_me(dataDir + 'tracking.json', 'w2v_list_gen.py')

    except Exception as e:
        print(repr(e))
        pdb.post_mortem()
    print('Operation complete')
