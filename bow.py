#! /NLPShare/nlpenv/bin/python3
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from common import *

#notes_root = dataDir + '/' + notesDir
feature_list = './features.txt'
num_folds = 5
ngram_range = (1, 2)
min_df = 50

def run_cross_validation(notesDirName):
  """Run n-fold CV and return average accuracy"""      

  notes_root = dataDir + notesDirName
  bunch = load_files(notes_root)
  pickleSave(bunch, dataDir + notesDirName + '_bunch3.obj')
  print('positive class:', bunch.target_names[1])
  print('negative class:', bunch.target_names[0])

  # raw occurences
  vectorizer = CountVectorizer(
    ngram_range=ngram_range, 
    stop_words='english',
    min_df=min_df ,
    vocabulary=None,
    binary=False)
  count_matrix = vectorizer.fit_transform(bunch.data)
  
  # print features to file for debugging
  feature_file = open(feature_list, 'w')
  for feature in vectorizer.get_feature_names():
    feature_file.write(feature + '\n')
  
  # tf-idf 
  tf = TfidfTransformer()
  tfidf_matrix = tf.fit_transform(count_matrix)

  x_train, x_test, y_train, y_test = train_test_split(
    tfidf_matrix, bunch.target, test_size = 0.2, random_state=0)
  pickleSave({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test},
             dataDir + notesDirName + '_train_test_split3.dct')
  clfs = []
  clfs.append(LinearSVC())
  clfs.append(LinearSVC(class_weight='balanced'))
  clfs += getClfs()
  pickleSave(clfs, dataDir + notesDirName + '_classifiers3.lst')

  for idx in range(len(clfs)):
    print('Processing classifier #%d\n' % idx)
    print('Classifier info: %s\n' % str(clfs[idx]))
    pred, prec, rec, f1 = classify(clfs[idx], x_train, y_train, x_test, y_test)
    print('Prediction: %s' % pred)
    #print('For #%d, precision is %s, recall is %s and f1 is %s' % (idx, str(pred), str(prec),
    #      str(rec), str(f1)))
    print('p =', prec)
    print('r = ', rec)
    print('f1 = ', f1)
    
  '''model = classifier.fit(x_train, y_train)
  predicted = classifier.predict(x_test)
  print 'predictions:', predicted
  print

  precision = precision_score(y_test, predicted, pos_label=1)
  recall = recall_score(y_test, predicted, pos_label=1)
  f1 = f1_score(y_test, predicted, pos_label=1)
  print 'p =', precision
  print 'r =', recall
  print 'f1 =', f1'''

def classify(classifier, x_train, y_train, x_test, y_test):

  try:
    model = classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    precision = precision_score(y_test, predicted, pos_label=1)
    recall = recall_score(y_test, predicted, pos_label=1)
    f1 = f1_score(y_test, predicted, pos_label=1)
    return predicted, precision, recall, f1

  except Exception as e:
    print('Error in classify. Attempting to ignore and recover...', e.args)
    return '**EXCEPTION**', -1, -1, -1

def getClfs():
  # make a bunch of classifiers
  clfs = []

  for C in [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]:

    for dual in [True, False]:

      for f_i in [True, False]:
        # fit_intercept

        for c_w in ['balanced', None]:
          # class_weight

          for loss in ['hinge', 'squared_hinge']:

            for penalty in ['l1', 'l2']:

              for tol in [1e-3, 1e-4]:

                try:
                  clfs.append(LinearSVC(C=C, class_weight=c_w, dual=dual, fit_intercept=f_i,
                                        loss=loss, penalty=penalty, tol=tol))

                except Exception as e:
                  print('Error in getClrs()', e.args)
  return clfs

if __name__ == "__main__":

  run_cross_validation('anc_notes')
