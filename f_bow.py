#! /home/aphillips5/envs/nlpenv/bin/python3

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm

from common import *

numCalls = 300  # number of calls; TODO: facilitate passing call num to called function
gSParams = [{'methods': ['gSGenericRunner']}, ['anc_notes_cuis', 'anc_notes_trim_v2_cuis'], ['LinearSVC'], [5],
            [(1,1), (1,2), (1,3), (2,2), (2,3)], [0, 5, 10, 50], ['l1', 'l2'], [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], ['balanced'], ['kf', 'tts'], [0]]  # grid search params
memo = {}  # for memoization

def gSGenericRunner(notesDirName, clf, numFolds, ngramRange, minDF, penalty, C, classWeight, modSel, randState):
  result = {}  # holds all the created objects, etc since pickleSave can't be used at this time
  tfidf_matrix, bunch = PreProc(notesDirName, ngramRange, minDF)
  clf = getattr(svm, clf)
  classifier = clf(penalty=penalty, C=C, class_weight=classWeight)
  result['classifier'] = str(classifier)

  try:
    #result['model'] = classifier.fit(x_train, y_train)
    #result['predicted'] = classifier.predict(x_test)
    #result['precision'] = precision_score(y_test, result['predicted'], pos_label=1)
    #result['recall'] = recall_score(y_test, result['predicted'], pos_label=1)
    #result['f1'] = f1_score(y_test, result['predicted'], pos_label=1)
    p = r = f1 = 0
    p, r, f1 = CrossVal(numFolds, classifier, tfidf_matrix, bunch) if modSel == 'kf' else TTS(randState, classifier, tfidf_matrix, bunch)
    result['precision'] = p
    result['recall'] = r
    result['f1'] = f1
    result['error'] = None

  except Exception as e:
    print('Error in classification. Attempting to ignore and recover...', e.args)
    result = {'classifier': result['classifier']}
    result['error'] = e.args
    result['f1'] = None
  print('Returning result for classifier %s: P = %s, R = %s, F1 = %s' % (result['classifier'], result['precision'], result['recall'], result['f1']))
  return result

def PreProc(notesDirName, ngramRange, minDF):
  # 17-07-07 preprocessing with memoization for better speed and efficient memory use
  param_hash = hash_sum('%s%s%d' % (notesDirName, str(ngramRange), minDF))
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

def CrossVal(numFolds, classifier, tfidf_matrix, bunch):
  # KFold
  ps = []
  rs = []
  f1s = []
  folds = KFold(n_splits=numFolds)

  for train_indices, test_indices in folds.split(tfidf_matrix):
    x_train = tfidf_matrix[train_indices]
    y_train = bunch.target[train_indices]
    x_test = tfidf_matrix[test_indices]
    y_test = bunch.target[test_indices]
    #classifier = sk.svm.LinearSVC()
    model = classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    ps.append(precision_score(y_test, pred, pos_label=1))
    rs.append(recall_score(y_test, pred, pos_label=1))
    f1s.append(f1_score(y_test, pred, pos_label=1))
  return np.mean(ps), np.mean(rs), np.mean(f1s)

def TTS(randState, classifier, tfidf_matrix, bunch):
  # train-test split
  x_train, x_test, y_train, y_test = train_test_split(
    tfidf_matrix, bunch.target, test_size=0.2, random_state=randState)
  #result['x_train'] = x_train
  #result['x_test'] = x_test
  #result['y_train'] = y_train
  #result['y_test'] = y_test
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
      results[idx] = 'Error in #%d: %s' % (idx, str(e))
      writeLog('%s: %s' % (currentTime(), str(e)))
  results.append(memo)
  saveJson(results, '%sexp%s_anc_notes_LSVC_CV_GS_results.json' % (dataDir, getExpNum(dataDir + 'tracking.json')))
  print('Operation complete.')
  e_time = currentTime()
  writeLog('Started %s and ended %s.' % (s_time, e_time))
