'''
17-08-03 - Rule based classifier
'''


import itertools, random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_score, recall_score, f1_score

from common import *

class RulesBasedClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, rules=None, thresh_cnt=1):
        # takes a list of regexes or function, threshold
        self._thresh_cnt = thresh_cnt
        if isinstance(rules, str) and os.path.exists(rules): rules = self._rules_from_file(rules)
        self._rules = rules if rules else self._default_rules()
        return

    def __str__(self):
        return 'RulesBasedClassifier(rules=%s, thresh_cnt=%d)' % (str(self._rules), self._thresh_cnt)

    def _default_rules(self):
        rules = ['intoxicat\w+\b', 'banana bag', '(alcohol|etoh).{1,20}(dependence|withdrawal|abuse)', 'drink.{1,45}(every|per|each|/).*(day|night|daily)', 'drunk', 'alcoholic', 'heavy (etoh|alcohol)', 'CRITICAL_BAC']
        return rules

    def _rules_from_file(self, path):
        # get from text or json file
        rules = None

        if path.endswith('.json'):
            content = loadJson(path)

            if isinstance(content, list):
                rules = content

        if path.endswith('.txt'):
            rules = loadText(path).split('\n')
        return rules

    def fit(self, X, y):
        # generate rule and sample stats
        self._X = X
        self._y = y
        re_stats = {}
        self._samp_stats = []

        for idx in range(len(X)):
            samp = X[idx]
            samp = '\n'.join(list(set(samp.split('\n'))))  # remove dups
            samp = samp.replace('\n', ' ')
            s_stat = [samp]

            for pat_idx in range(len(self._rules)):
                pat = self._rules[pat_idx]
                if not pat in re_stats: re_stats[pat] = [1] * 4  # rfp, sfp, rfn, sfn
                match_ctr = len(re.findall(pat, samp))
                if match_ctr > 0:
                    s_stat.append([pat, match_ctr])
                    if y[idx] == 1: re_stats[pat][0] += match_ctr; re_stats[pat][1] += 1
                    if y[idx] == 0: re_stats[pat][2] += match_ctr; re_stats[pat][3] += 1
            self._samp_stats.append(s_stat)
        self._re_stats = re_stats
        return self

    def predict(self, X):
        preds = []
        try:
            if not isinstance(self._rules, list): return self._rules()

        except Exception as e:
            print(repr(e))
            pdb.post_mortem()

        for samp in X:
            result = [1] * 4

            for pat in self._re_stats:
                match_ctr = len(re.findall(pat, samp))
                if match_ctr > 0:
                    stats = self._re_stats[pat]
                    result[0] += stats[0] / stats[1]  # rf_pos/sf_pos
                    result[1] += stats[2] / stats[3]  # rf_neg/sf_neg
            result[2] = result[0] + result[1]
            result[3] = result[0] - result[1]
            #pdb.set_trace()

            if result[0] / result[1] > 1:
                preds.append(1)

            else:
                preds.append(0)
        return preds

    def predict_1(self, X):
        preds = []
        try:
            if not isinstance(self._rules, list): return self._rules()

        except Exception as e:
            print(repr(e))
            pdb.post_mortem()

        for samp in X:
            match_ctr = 0

            for pat in self._rules:
                if re.search(r'%s' % pat, samp): match_ctr += 1
            if match_ctr >= self._thresh_cnt:
                preds.append(1)

            else:
                preds.append(0)
        return preds

class OptimizedRulesSeeker(BaseEstimator, ClassifierMixin):

    def __init__(self, search=1, rand_state=0, optimum='f1'):
        self._search = search  # %age of samples to get rules from
        self._rand_state = rand_state  # seed to randomize search
        self._optimum = optimum  # score to optimize for

    def __str__(self):
        return 'OptimizedRulesSeeker(search=%d, rand_state=%d, optimum=%s)' % (self._search, self._rand_state, self._optimum)

    def fit(self, X, y):
        try:
            self._X = X
            self._y = y
            regexes = self._generate_regexes()
            self._clf = RulesBasedClassifier(regexes).fit(X, y)

        except Exception as e:
            print(repr(e))
            pdb.post_mortem()
        return self

    def _generate_regexes(self):
        word_tok = r'((\b[a-zA-Z0-9_/\-&]+\b)|(\b\(d*,?d+)+(\.?\d+)?%?\w*\*?\b))'
        word_sep = re.compile(r'[\s;.]')
        sent_sep = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")
        regexes = []
        targ_samps = self._set_targs(self._y, self._rand_state, self._search)

        for idx in targ_samps:
            samp = self._X[idx]
            samp = samp.replace('\n', ' ')
            sents = sent_sep.split(samp)

            for sent in sents:
                words = word_sep.split(sent)
                regexes += self._res_from_words(words, word_tok)
        regexes =  list(set(regexes))
        writeLog('%s: %d regexes generated from %d samples!' % (currentTime(), len(regexes), len(targ_samps)))
        return regexes

    def _res_from_words(self, words, tok):
        targ_len = len(words) - 2
        combs = []  # word combos
        res = []  # regexes

        for idx in range(targ_len+3):
            # create combinations of all words in the sentence
            elems = [list(x) for x in itertools.combinations(words, idx)]
            combs.extend(elems)

        for idx in range(len(combs)):
            # make each combo into a regex

            if len(combs[idx]) == 1:
                res.append(combs[idx][0])
                continue
            max_words_btwn = targ_len - len(combs[idx]) + 3
            btwn_re = r'%s{,%d}' % (tok, max_words_btwn)
            res.append(btwn_re.join(combs[idx]))
        return res

    def _set_targs(self, targ, rand, search):
        f_label = targ[0]
        s_label = None
        split_pos = None
        if not isinstance(rand, int): rand = 0

        for idx in range(len(targ)):
            if targ[idx] == f_label: continue
            s_label = targ[idx]
            split_pos = idx
        f_size = int(search / 100 * len(targ[:split_pos]))
        s_size = int(search / 100 * len(targ[split_pos:]))
        f_rng = list(range(f_size))
        s_rng = list(range(split_pos, split_pos + s_size))
        if rand == 0: return f_rng + s_rng
        random.seed(rand)  # deterministic random state
        return random.sample(range(split_pos), f_size) + random.sample(range(split_pos, len(targ)), s_size)

    def predict(self, X):
        return self._clf.predict(X)

if __name__ == '__main__':
    print('This module contains importable classifiers')
#pdb.set_trace()
commit_me(dataDir + 'tracking.json', 'custom_clfs.py')