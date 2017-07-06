#! /home/aphillips5/envs/nlpenv/bin/python3

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm

from common import *

numCalls = 300  # number of calls; TODO: facilitate passing call num to called function
gSParams = [{'methods': ['gSGenericRunner']}, ['anc_notes_trim_v2_cuis', 'anc_notes_trim_cuis'], ['LinearSVC'], [5],
            [(1,1), (1,2), (1,3), (2,2), (2,3)], [5, 10, 50], ['l1', 'l2'], [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], ['balanced', None]]  # grid search params

def gSGenericRunner(notesDirName, clf, numFolds, ngramRange, minDF, penalty, C, classWeight):
  result = {}  # holds all the created objects, etc since pickleSave can't be used at this time
  notesRoot = dataDir + notesDirName
  bunch = load_files(notesRoot)
  #result['bunch'] = bunch
  #print('positive class:', bunch.target_names[1])
  #print('negative class:', bunch.target_names[0])

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

  x_train, x_test, y_train, y_test = train_test_split(
    tfidf_matrix, bunch.target, test_size = 0.2, random_state=0)
  #result['x_train'] = x_train
  #result['x_test'] = x_test
  #result['y_train'] = y_train
  #result['y_test'] = y_test
  clf = getattr(svm, clf)
  classifier = clf(penalty=penalty, C=C, class_weight=classWeight)
  result['classifier'] = str(classifier)

  try:
    result['model'] = classifier.fit(x_train, y_train)
    result['predicted'] = classifier.predict(x_test)
    result['precision'] = precision_score(y_test, result['predicted'], pos_label=1)
    result['recall'] = recall_score(y_test, result['predicted'], pos_label=1)
    result['f1'] = f1_score(y_test, result['predicted'], pos_label=1)
    result['error'] = None

  except Exception as e:
    print('Error in classification. Attempting to ignore and recover...', e.args)
    result = {'classifier': result['classifier']}
    result['error'] = e.args
    result['f1'] = None
  print('Returning result for classifier %s: P = %s, R = %s, F1 = %s' % (result['classifier'], result['precision'], result['recall'], result['f1']))
  return result


if __name__ == "__main__":
  results = gridSearchAOR(gSParams, doEval=False)
  print('First and last method calls:', results[0], results[-1])

  for idx in range(len(results)):

    try:
      print('Processing #%d of %d: %s' % (idx + 1, len(results), results[idx]))
      results[idx] = [results[idx], eval(results[idx])]

    except Exception as e:
      results[idx] = 'Error in #%d: %s' % (idx, str(e))
      print(str(e))
  saveJson(results, '%sexp%s_anc_notes_LSVC_GS_results.json' % (dataDir, getExpNum(dataDir + 'tracking.json')))
  print('Operation complete.')
