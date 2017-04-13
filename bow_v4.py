#! /NLPShare/nlpenv/bin/python3

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import naive_bayes

from common import *

numCalls = 300  # number of calls; TODO: facilitate passing call num to called function
gSParams = [{'methods': ['gSGenericRunner']}, ['anc_notes_trim'], ['MultinomialNB', 'BernoulliNB'], [5],
            [(1,1), (1,2), (1,3), (2,2), (2,3)], [5, 10, 50]]  # grid search params

def gSGenericRunner(notesDirName, clf, numFolds, ngramRange, minDF):
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
  clf = getattr(naive_bayes, clf)
  classifier = clf()
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
      print('Processing ' + results[idx])
      results[idx] = [results[idx], eval(results[idx])]

    except Exception as e:
      results[idx] = 'Error in #%d: %s' % (idx, str(e))
      print(str(e))
  pickleSave(results, '%sanc_notes_trim_NB_GS_exp%s_results.lst' % (dataDir, getExpNum(dataDir + 'tracking.dct')))
  print('Operation complete.')
