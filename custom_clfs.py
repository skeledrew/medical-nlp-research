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
        writeLog('%s: Generating regexes from %d samples...' % (currentTime(), len(targ_samps)))

        for idx in targ_samps:
            samp = self._X[idx]
            samp = samp.replace('\n', ' ')
            sents = sent_sep.split(samp)

            for sent in sents:
                words = word_sep.split(sent)
                regexes += self._res_from_words(words, word_tok)
        regexes =  list(set(regexes))
        writeLog('%s: %d regexes generated!' % (currentTime(), len(regexes), len(targ_samps)))
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

class BitVectorizor():

    def __init__(self, ngram_range=[1,1], min_df=None, splits=[r'\. ', r' ']):
        self._splits = splits
        self._black_list = []
        self._ent_list = []
        self._idx_vec_list = []
        self._bin_vec_list = []
        #self._split_level_cnt = len(splits)
        self._tmp_doc = []
        self._bit_matrix = []
        self._ent_list_pos = 0

    def fit_transform(self, docs):
        if not isinstance(docs, list): raise ValueError('BitVectorizor can only transform a list of documents')  # TODO: ensure this is a list and not array
        blob = []
        #pdb.set_trace()

        for doc in docs:
            # break into snippets of words, sentences, etc
            if not isinstance(doc, str): raise ValueError('Doc must be a string')
            blob.append(self.split_doc(doc.lower(), splits=self._splits[:]))
            self._make_numbers()
            self._make_bits()
            self._bit_matrix.append(self._tmp_doc)
            self._tmp_doc = []
            self._ent_list_pos = len(self._ent_list)
        max_len = len(self._bit_matrix[-1])

        for idx, bv in enumerate(self._bit_matrix):
            # make all bit vectors the same length
            self._bit_matrix[idx] = int(bv.ljust(max_len, '0'), 2)
        return self._bit_matrix

    def _make_numbers(self):

        for idx in range(self._ent_list_pos, len(self._ent_list)):
            # process each snippet entity
            ent = self._ent_list[idx]

            if isinstance(ent, str):
                # single word
                ent_idx = self._ent_list.index(ent)
                if ent_idx in self._idx_vec_list: continue
                self._idx_vec_list.append(ent_idx)
                self._tmp_doc.append(ent_idx)
                continue
            if not isinstance(ent, list): raise ValueError('"{}" must be a string or list'.format(ent))
            idx_vec = []
            ## TODO: try and detect if the entity was already vec'd

            for sub in ent:
                # list of entities
                if not sub in self._ent_list: raise ValueError('"{}" is not in the list of known entities'.format(sub))
                idx_vec.append(self._ent_list.index(sub))
            idx_vec.sort()
            if not idx_vec in self._idx_vec_list: self._idx_vec_list.append(idx_vec)
            self._tmp_doc.append(idx_vec)
        return

    def _make_bits(self):
        # TODO: need to reverse and append zeros after all docs read

        #for idx in range(self._ent_list_pos, len(self._idx_vec_list)):
            #
        bit_vec = ['0'] * len(self._idx_vec_list)

        for number in self._tmp_doc:
            bit_vec[self._idx_vec_list.index(number)] = '1'
        self._tmp_doc = ''.join(bit_vec[::-1])
        return

    def split_doc(self, doc, splits=[' ']):
        '''Recursively splits a snippet'''

        if not splits:
            # word level
            if not doc in self._ent_list: self._ent_list.append(doc)
            #if not doc in self._tmp_doc: self._tmp_doc.append(doc)
            return doc
        split = splits[0]
        s_doc = re.split(split, doc) if isinstance(split, str) else self.make_ngrams(doc, split)

        for idx, part in enumerate(s_doc):
            s_doc[idx] = self.split_doc(part, splits[1:]) if splits else s_doc[idx]
            if isinstance(s_doc[idx], list): s_doc[idx].sort()
            if not s_doc[idx] in self._ent_list: self._ent_list.append(s_doc[idx])
            #if not s_doc[idx] in self._tmp_doc: self._tmp_doc.append(s_doc[idx])
        splits.pop(0)
        return s_doc

    def make_ngrams(doc, ng_range):
        '''Extract a range of given ngrams'''
        ng_doc = []
        if not (type(ng_range) in [list, tuple] or len(ng_range)): raise ValueError('N-gram range must be an iterable of 2 elements')

        for ngram in range(ng_range[0], ng_range[1]):

            for idx, _ in enumerate(doc):

                try:
                    ng_doc.append(doc[idx:idx + ngram])

                except:
                    # reached out of range
                    break
        return doc

class BitMappingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, algo='simple', iteration=50, rand_state=1, class_weight='balanced', sample_size=10, tolerance=15, stop_words=['']):
        self._algo = algo  # str
        self._iter = iteration  # int
        self._weight = class_weight  # str
        self._samp_size = sample_size  # %age
        self._rand = rand_state  # int
        self._tol = tolerance  # %age; use for black listing feats
        self._classes = {}
        self._algos = {}
        self._stop_words = []
        self._register_stops(stop_words)
        self._load_algos()

    def _load_algos(self):
        self._algos['simple'] = self._algo_simple_

    def _register_stops(sws):
        if not sws: return
        if isinstance(sws, str) and exists(sws): sws = open(sws).read()
        if isinstance(sws, str) and '[' in sws and ']' in sws: sws = json.loads(sws)
        self._stop_words = sws

    def fit(self, X, y):
        '''Train model'''
        #if DEBUG: pdb.set_trace()

        for doc, lbl in zip(X, y):
            # sort docs by class
            lbl = str(lbl)
            if not lbl in self._classes: self._classes[lbl] = []
            self._classes[lbl].append(doc)
        samps = {}

        for lbl in self._classes:
            # get samples
            samps[lbl] = random.sample(self._classes[lbl], self._samp_size)
        class_prints = {}

        for lbl in samps:
            # create "class prints"
            class_print = ''.join(['0'] * (len(bin(samps[lbl][0]))-2))
            bit_sum = 0

            for samp in samps[lbl]:
                # bitwise OR with each class sample
                bit_sum |= samp
            class_prints[bit_sum] = lbl  # flipped it; dunno why :/
        #black_list = []
        self._class_prints = class_prints
        return self

    def predict(self, X):
        '''Get predictions for a set of documents'''
        preds = []

        for doc in X:
            pred = self.run_algo(doc)
            preds.append(int(pred))
        return preds

    def run_algo(self, doc, algo=None):
        if not algo: algo = self._algo
        pred = None

        try:
            pred = self._algos.get(algo, 'simple')(doc)

        except Exception as e:
            print('Something broke:', repr(e))
            pass
        return pred

    def _algo_simple_(self, doc):
        last_diff = -1
        last_class = None
        #if DEBUG: pdb.set_trace()

        for bs in self._class_prints:
            curr_diff = self.count_set_bits(doc & bs)
            if not curr_diff > last_diff: continue
            last_diff = curr_diff
            last_class = self._class_prints[bs]
        if type(last_class) == type(None): pdb.set_trace()
        return last_class

    def count_set_bits(self, val):
        return bin(val).count('1')

class BitKNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_neighbors=5, rand_state=None, samp_size=100, max_diff=None, min_same=None):
        self._n_nei = n_neighbors
        self._rand_state = rand_state
        self._classes = {}
        self._samp_size = samp_size
        self._max_diff = max_diff
        self._min_same = min_same

    def fit(self, X, y):

        for doc, lbl in zip(X, y):
            lbl = str(lbl)
            if not lbl in self._classes: self._classes[lbl] = []
            self._classes[lbl].append(doc)
        samps = {}
        if isinstance(self._rand_state, int): random.seed(self._rand_state)

        for lbl in self._classes:
            samps[lbl] = random.sample(self._classes[lbl], self._samp_size)
            self._samps = samps
        return self

    def predict(self, X):
        preds = []

        for doc in X:
            pred = self.do_knn(doc)
            preds.append(int(pred))
        return preds

    def do_knn(self, doc):
        # bit diff KNN
        scores = {}
        totals = {}
        max_nearest = None

        for lbl in self._samps:
            # get all diffs together
            if not lbl in scores:
                scores[lbl] = []
                totals[lbl] = 0

            for samp in self._samps[lbl]:
                diff = bin(doc & samp).count('1')  # WARN: breaks DRY
                scores[lbl].append(diff)
            scores[lbl].sort()

        for idx in range(self._n_nei):
            # find class with most smallest diffs
            curr_max = None
            curr_score = None

            for lbl in scores:
                # find current smallest diff
                if curr_max == None or scores[lbl][idx] < curr_score: curr_max = lbl; curr_score = scores[lbl][idx]
            totals[curr_max] += 1
        best_score = None
        best_class = None

        for lbl in totals:
            # get the class with the most nearest
            if not best_class or best_score < totals[lbl]: best_class = lbl; best_score = totals[lbl]
        return best_class

DEBUG = True

if __name__ == '__main__':
    print('This module contains importable classifiers')
#pdb.set_trace()
commit_me(dataDir + 'tracking.json', 'custom_clfs.py')
