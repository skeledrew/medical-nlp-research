'''
17-08-03 - Rule based classifier
'''


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

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
        rules = ['intoxicat\w+\b', 'banana bag', '(alcohol|etoh).+(dependence|withdrawal)', 'drink.{1,45}(every|per|each|/).*(day|night)', 'drunk', 'alcoholic', 'heavy (etoh|alcohol)']
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
        self._X = X
        self._y = y
        return self

    def predict(self, X):
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

if __name__ == '__main__':
    print('This module contains importable classifiers')
commit_me(dataDir + 'tracking.json')
