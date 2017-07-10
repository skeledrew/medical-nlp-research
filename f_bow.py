#! /home/aphillips5/envs/nlpenv/bin/python3

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm, naive_bayes, linear_model, neighbors

from common import *


IGNORE = '~~IGNORE_THIS_PARAM~~'
numCalls = 300  # number of calls; TODO: facilitate passing call num to called function
gSParams = [
  {'methods': ['gSGenericRunner']},
  ['anc_notes_trim', 'anc_notes_trim_v2_cuis'],  # data dirs
  [
    'LinearSVC',
    'BernoulliNB',
    'SVC',
    'Perceptron',
    'SGDClassifier',
    'LogisticRegression',
    'PassiveAggressiveClassifier',
    'NearestCentroid',
    'KNeighborsClassifier'
  ],  # classifiers
  [5],  # for n-folds CV
  [(1,1), (1,2), (1,3), (2,2), (2,3), (3,4)],  # n-grams
  [0, 5, 10, 50],  # minDF
  [None, 'l1', 'l2', 'elasticnet'],  # penalty
  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],  # C
  ['balanced'],  # weight
  ['kf', 'tts'],  # validation method
  [0],  # TTS random state
  [
    'hinge',
    'log',
    #'modified_huber',
    'squared_hinge',
    #'perceptron',
    #'squared_loss',
    #'huber',
    #'epsilon_insensitive',
    #'squared_epsilon_insensitive'
  ],  # loss
  [5], # n_neighbors
  ['uniform', 'distance'], # KNN weights
]  # grid search params
memo = {}  # for memoization
clfMods = [svm, naive_bayes, linear_model, neighbors]

def gSGenericRunner(
    notesDirName,
    clfName,
    numFolds,
    ngramRange,
    minDF,
    penalty,
    C,
    classWeight,
    modSel,
    randState,
    loss,
    nNei,
    kNWeights,
):
  result = {}  # holds all the created objects, etc
  frame = currentframe()
  args, _, _, values = getargvalues(frame)
  result['options'] = {arg: values[arg] for arg in args}
  preproc_hash = hash_sum('%s%s%d' % (notesDirName, str(ngramRange), minDF))
  tfidf_matrix, bunch = PreProc(notesDirName, ngramRange, minDF, preproc_hash)
  hyParams = {
    'penalty': penalty,
    'C': C,
    'class_weight': classWeight,
    'loss': loss,
    'n_neighbors': nNei,
    'weights': kNWeights
  }
  classifier = MakeClf(clfName, hyParams, clfMods)
  result['classifier'] = str(classifier)
  clf_hash = hash_sum(result['classifier'])
  if not clf_hash in memo: memo[clf_hash] = classifier

  try:
    p = r = f1 = 0
    p, r, f1 = CrossVal(numFolds, classifier, tfidf_matrix, bunch, preproc_hash, clf_hash) if modSel == 'kf' else TTS(randState, classifier, tfidf_matrix, bunch, preproc_hash, clf_hash)
    result['precision'] = p
    result['recall'] = r
    result['f1'] = f1
    result['error'] = None

  except Exception as e:
    print('Error in classification. Attempting to ignore and recover...', e.args)
    result = {'classifier': result['classifier'], 'options': result['options']}
    result['error'] = e.args
    result['f1'] = result['precision'] = result['recall'] = None
  print('Classifier %s \nwith options %s yielded: P = %s, R = %s, F1 = %s' % (result['classifier'], str(result['options']), result['precision'], result['recall'], result['f1']))
  return result

def PreProc(notesDirName, ngramRange, minDF, param_hash):
  # 17-07-07 preprocessing with memoization for better speed and efficient memory use
  if param_hash in memo: return memo[param_hash]['tfidf_matrix'], memo[param_hash]['bunch']
  notesRoot = dataDir + notesDirName
  b_hash = hash_sum(notesRoot)
  bunch = memo[b_hash] if b_hash in memo else load_files(notesRoot)
  if not b_hash in memo: memo[b_hash] = bunch

  # raw occurences
  vectorizer = CountVectorizer(
    ngram_range=ngramRange,
    stop_words='english',
    min_df=minDF,
    vocabulary=None,
    binary=False)
  count_matrix = vectorizer.fit_transform(bunch.data)

  # print features to file for debugging
  '''feature_file = open(feature_list, 'w')
  for feature in vectorizer.get_feature_names():
    feature_file.write(feature + '\n')'''

  # tf-idf
  tf = TfidfTransformer()
  tfidf_matrix = tf.fit_transform(count_matrix)
  memo[param_hash] = {}
  memo[param_hash]['tfidf_matrix'] = tfidf_matrix
  memo[param_hash]['bunch'] = bunch
  return tfidf_matrix, bunch

def MakeClf(clf_name, hyparams, clf_mods):
  # create classifier and add relevant hyperparameters
  clf = None
  mod = None

  for m in clf_mods:
    # get the module containing the classifier
    # NB: doesn't handle possible duplicate class names in diff mods
    if not clf_name in m.__dict__: continue
    mod = m
    break
  if not mod: raise Exception('Unable to find "%s" in any of the given modules.' % clf_name)
  clf = getattr(mod, clf_name)
  clf_str = str(clf())
  params = ', '.join(['%s=%s' % (p, hyparams[p] if not isinstance(hyparams[p], str) else hyparams[p].__repr__()) for p in hyparams if p + '=' in clf_str and not hyparams[p] == IGNORE])  # make parameter string
  classifier = eval('clf(%s)' % (params))
  return classifier

def CrossVal(numFolds, classifier, tfidf_matrix, bunch, pp_hash, clf_hash):
  # KFold
  kf_hash = hash_sum('%d%s%s' % (numFolds, pp_hash, clf_hash))
  if kf_hash in memo: raise Exception('Arg combo already processed. Skipping...')
  memo[kf_hash] = True
  ps = []
  rs = []
  f1s = []
  folds = KFold(n_splits=numFolds)

  for train_indices, test_indices in folds.split(tfidf_matrix):
    x_train = tfidf_matrix[train_indices]
    y_train = bunch.target[train_indices]
    x_test = tfidf_matrix[test_indices]
    y_test = bunch.target[test_indices]
    model = classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    ps.append(precision_score(y_test, pred, pos_label=1))
    rs.append(recall_score(y_test, pred, pos_label=1))
    f1s.append(f1_score(y_test, pred, pos_label=1))
  return np.mean(ps), np.mean(rs), np.mean(f1s)

def TTS(randState, classifier, tfidf_matrix, bunch, pp_hash, clf_hash):
  # train-test split
  tts_hash = hash_sum('%d%s%s' % (randState, pp_hash, clf_hash))
  if tts_hash in memo: raise Exception('Arg combo already processed. Skipping...')
  memo[tts_hash] = True
  x_train, x_test, y_train, y_test = train_test_split(
    tfidf_matrix, bunch.target, test_size=0.2, random_state=randState)
  model = classifier.fit(x_train, y_train)
  pred = classifier.predict(x_test)
  p = precision_score(y_test, pred, pos_label=1)
  r = recall_score(y_test, pred, pos_label=1)
  f1 = f1_score(y_test, pred, pos_label=1)
  return p, r, f1

if __name__ == "__main__":
  s_time = currentTime()
  results = gridSearchAOR(gSParams, doEval=False)
  print('First and last method calls:', results[0], results[-1])

  for idx in range(len(results)):

    try:
      writeLog('%s: Processing #%d of %d: %s' % (currentTime(), idx + 1, len(results), results[idx]))
      results[idx] = [results[idx], eval(results[idx])]

    except Exception as e:
      #raise Exception(e)
      results[idx] = 'Error in #%d: %s' % (idx, str(e))
      writeLog('%s: %s' % (currentTime(), str(e)))
  results.append(gSParams)
  results.append(memo)
  saveJson(results, '%sexp%s_anc_notes_GS_results.json' % (dataDir, getExpNum(dataDir + 'tracking.json')))
  print('Operation complete.')
  e_time = currentTime()
  writeLog('Started %s and ended %s.' % (s_time, e_time))
