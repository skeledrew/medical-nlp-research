#! /NLPShare/nlpenv/bin/python3

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm

from common import *

#feature_list = './features.txt'
#num_folds = 5
#ngram_range = (1, 2)
#min_df = 50
numCalls = 300  # number of calls; TODO: facilitate passing call num to called function
gSParams = [{'methods': ['gSGenericRunner']}, ['anc_notes_trim'], [5],
            [(1,1), (1,2), (1,3), (2,2), (2,3)], [5, 10, 50], [1, 10, 100, 1000],
            ['rbf', 'poly'], [2, 3], ['balanced'], ['LinearSVC', 'SVC']]  # grid search params

def gSGenericRunner(notesDirName, numFolds, ngramRange, minDF, C, kernel, degree, classWeight, clf):
  result = {}  # holds all the created objects, etc since pickleSave can't be used at this time
  notesRoot = dataDir + notesDirName
  bunch = load_files(notesRoot)
  #pickleSave(bunch, '%s%s_bunch%s.obj' % (dataDir, notesDirName, version))
  result['bunch'] = bunch
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
  #pickleSave({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test},
  #           dataDir + notesDirName + '_train_test_split3.dct')
  result['x_train'] = x_train
  result['x_test'] = x_test
  result['y_train'] = y_train
  result['y_test'] = y_test
  clf = getattr(svm, clf)
  classifier = clf(C=C, kernel=kernel, degree=degree, class_weight=classWeight)
  result['classifier'] = str(classifier)
  #pickleSave(clfs, dataDir + notesDirName + '_classifiers3.lst')

  try:
    result['model'] = classifier.fit(x_train, y_train)
    result['predicted'] = classifier.predict(x_test)
    result['precision'] = precision_score(y_test, result['predicted'], pos_label=1)
    result['recall'] = recall_score(y_test, result['predicted'], pos_label=1)
    result['f1'] = f1_score(y_test, result['predicted'], pos_label=1)
    result['error'] = None

  except Exception as e:
    print('Error in classification. Attempting to ignore and recover...', e.args)
    result['error'] = e.args
  print('Returning result for classifier %s, F1 = %s' % (result['classifier'], result['f1']))
  return result

if __name__ == "__main__":
  results = gridSearchAOR(gSParams, doEval=False)
  print('First and last method calls:', results[0], results[-1])

  for idx in range(len(results)):

    try:
      results[idx] = [results[idx], eval(results[idx])]

    except Exception as e:
      results[idx] = [results[idx], 'Error in #%d: %s' % (idx, str(e))]
  pickleSave(results, '%sanc_notes_trim_SVC_GS_exp%s_results.lst' % (dataDir, getExpNum(dataDir + 'tracking.dct')))
  print('Operation complete.')
