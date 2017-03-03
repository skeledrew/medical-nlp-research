#! /NLPShare/nlpenv/bin/python3


'''
Bag of Words with SciKit-Learn
- Original copied from Fall 2016 ARDS archive
- Modified to use alcohol data and pickle ALL THE THINGS
'''


import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from common import *

#notes_root = dataDir + 'anc_notes/'
#feature_list = './features.txt'
num_folds = 10
ngram_range = (1, 3)
min_df = 50
max_df = 2000

def run_cross_validation(noteDirName):
  """Run n-fold CV and return average accuracy"""      

  notes_root = dataDir + noteDirName
  feature_list = dataDir + noteDirName + '_features.txt'
  bunch = load_files(notes_root)
  pickleSave(bunch, dataDir + noteDirName + '_bunch.obj')
  print ('positive class:', bunch.target_names[1])
  print ('negative class:', bunch.target_names[0])

  # raw occurences
  vectorizer = CountVectorizer(
    ngram_range=ngram_range, 
    stop_words='english',
    max_df=max_df,
    min_df=min_df ,
    vocabulary=None,
    binary=False)
  pickleSave(vectorizer, dataDir + noteDirName + '_vectorizer.obj')
  count_matrix = vectorizer.fit_transform(bunch.data)
  
  # print features to file for debugging
  feature_file = open(feature_list, 'w')
  for feature in vectorizer.get_feature_names():
    feature_file.write(feature + '\n')
  
  # tf-idf 
  tf = TfidfTransformer()
  tfidf_matrix = tf.fit_transform(count_matrix)
  pickleSave(tfidf_matrix, dataDir + noteDirName + '_tfidf_matrix.obj')

  x_train, x_test, y_train, y_test = train_test_split(
    tfidf_matrix, bunch.target, test_size = 0.1, random_state=0)
  pickleSave({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test},
             dataDir + noteDirName + '_train_test_split.dct')
 
  # grid search
  params = [{'kernel':['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel':['linear'], 'C': [1, 10, 100, 1000]}]

  scores = ['precision', 'recall']

  for score in scores:
      print("Hyperparameter tuning for %s score\n" % score)

      classifier = GridSearchCV(estimator=SVC(), param_grid=params, scoring = '%s_macro' % score, cv=num_folds)
      
      classifier.fit(x_train, y_train)
      pickleSave(classifier, dataDir + noteDirName + '_' + score + '_GridSearchCV.obj')
      print(classifier.best_params_)


  #***ORIGINAL IMPLEMENTATION***# 
  #classifier = LinearSVC(class_weight='balanced')
  #model = classifier.fit(x_train, y_train)
  #predicted = classifier.predict(x_test)
  #print ('predictions:', predicted)
  #print

  #precision = precision_score(y_test, predicted, pos_label=1)
  #recall = recall_score(y_test, predicted, pos_label=1)
  #f1 = f1_score(y_test, predicted, pos_label=1)
  #print ('p =', precision)
  #print ('r =', recall)
  #print ('f1 =', f1)  

if __name__ == "__main__":

  run_cross_validation('lab_notes')
  run_cross_validation('other_notes')
