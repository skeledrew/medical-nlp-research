#! /home/aphillips5/envs/nlpenv/bin/python3

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm, naive_bayes, linear_model, neighbors, ensemble

from common import *
import custom_clfs


DEBUG = False
IGNORE = '~~IGNORE_THIS_PARAM~~'
numCalls = 300  # number of calls; TODO: facilitate passing call num to called function
gSParams = [
  {'methods': ['gSGenericRunner']},
  [
    #'anc_notes',  # complete notes
    #'anc_notes_trim',  # peel.py applied
    #'anc_notes_cuis',  # cuis w/out dict fix
    #'anc_notes_trim_cuis',
    #'anc_notes_v2_cuis',  # cuis w/ dict fix
    #'anc_notes_trim_v2_cuis',  # trim cuis
    #'anc_bac-yn',  # BAC y/n values, from anc but no notes
    'anc_notes_trim_v3',  # trim with BAC
    'anc_notes_trim_v3_cuis',  # trim cuis with BAC
    #'anc_notes_trim_bac-all'
  ],  # data dirs
  [
    #'LinearSVC',
    #'BernoulliNB',
    #'SVC',
    ##'Perceptron',  # NB: Perceptron() is equivalent to SGDClassifier(loss=”perceptron”, eta0=1, learning_rate=”constant”, penalty=None)
    #'SGDClassifier',
    #'LogisticRegression',
    #'PassiveAggressiveClassifier',
    #'NearestCentroid',
    #'KNeighborsClassifier',
    #'MultinomialNB',
    #'GaussianNB'
    #'PassiveAggressiveRegressor',
    #'SGDRegressor',
    #'RulesBasedClassifier',  # custom
    'RandomForestClassifier',
  ],  # classifiers
  [10],  # for n-folds CV
  [
    (1,1),
    #(1,2),
    #(1,3),
    #(2,2),
    #(2,3)
  ],  # n-grams
  [
    0,
    #10,
    #50
  ],  # minDF
  [
    #None,
    'l1',
    #'l2',
    #'elasticnet',
    #'none'
  ],  # penalty
  [
    #0.0000001,
    #0.000001,
    #0.00001,
    #0.0001,
    #0.001,
    #0.01,
    #0.1,
    1,
    #10,
    #100,
    #1000,
    #10000,
    #100000,
    #1000000,
    #10000000,
  ],  # C
  ['balanced'],  # weight
  [
    'kf',
    #'tts'
  ],  # validation method
  [0],  # TTS random state
  [
    #'hinge',
    #'log',
    #'modified_huber',
    #'squared_hinge',
    #'perceptron',
    #'squared_loss',
    'huber',
    #'epsilon_insensitive',
    #'squared_epsilon_insensitive',
  ],  # loss
  [5], # n_neighbors
  [
    'uniform',
    #'distance'
  ], # KNN weights
  [
    #'constant',
    'optimal',
    #'invscaling'
  ],  # SGD learning rate
  [
    #'rbf',
    'linear',
    #'poly',
    #'sigmoid'
  ],  # kernel
  [
    'word',
    #'char_wb',
  ],  # CVec analyzer
  [
    #True,
    False
  ],  # CVec binary
  [
    #'text',
    'count',
    'tfidf',
  ],  # preprocessing task
  [
    -1,  # all CPUs
    #1,
  ],  # n jobs
]  # grid search params
memo = {}  # for memoization
clfMods = [svm, naive_bayes, linear_model, neighbors, custom_clfs, ensemble]

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
    learnRate,
    kernel,
    analyzer,
    binary,
    preTask,
    nJobs,
):
  result = {}  # holds all the created objects, etc
  frame = currentframe()
  args, _, _, values = getargvalues(frame)
  result['options'] = allArgs = {arg: values[arg] for arg in args}
  preproc_hash = hash_sum('%s%s%d%s%s' % (notesDirName, str(ngramRange), minDF, analyzer, binary))
  matrix, bunch, result['features'] = PreProc(notesDirName, ngramRange, minDF, analyzer, binary, preTask, preproc_hash)
  hyParams = {
    'penalty': penalty,
    'C': C,
    'class_weight': classWeight,
    'loss': loss,
    'n_neighbors': nNei,
    'weights': kNWeights,
    'learning_rate': learnRate,
    'kernel': kernel,
    'n_jobs': nJobs,
  }
  classifier = MakeClf(clfName, hyParams, clfMods)
  result['classifier'] = str(classifier).replace('\n', '')
  clf_hash = hash_sum(result['classifier'])
  if not clf_hash in memo: memo[clf_hash] = classifier

  try:
    p = r = f1 = std = 0
    p, r, f1, std, mis = CrossVal(numFolds, classifier, matrix, bunch, preproc_hash, clf_hash) if modSel == 'kf' else TTS(randState, classifier, matrix, bunch, preproc_hash, clf_hash)
    result['precision'] = p
    result['recall'] = r
    result['f1'] = f1
    result['std'] = std
    result['mis'] = mis
    result['error'] = None

  except Exception as e:
    writeLog('%s: Error in classification: %s. Skipping...' % (currentTime(), repr(e)))
    result = {'classifier': result['classifier'], 'options': result['options']}
    result['error'] = e.args
    result['f1'] = result['precision'] = result['recall'] = result['std'] = None
  if result['f1']: writeLog('%s: Classifier %s \nwith options %s yielded: P = %s, R = %s, F1 = %s, Std = %s' % (currentTime(), result['classifier'], str(result['options']), result['precision'], result['recall'], result['f1'], result['std']))
  return result

def PreProc(notesDirName, ngramRange, minDF, analyzer, binary, pre_task, param_hash):
  # 17-07-07 preprocessing with memoization for better speed and efficient memory use
  if param_hash in memo: return memo[param_hash]['matrix'], memo[param_hash]['bunch'], memo[param_hash]['features']
  memo[param_hash] = {}
  notesRoot = dataDir + notesDirName
  b_hash = hash_sum(notesRoot)
  bunch = memo[b_hash] if b_hash in memo else load_files(notesRoot)
  if not b_hash in memo: memo[b_hash] = bunch
  memo[param_hash]['bunch'] = bunch

  if pre_task == 'text':
    text_matrix = np.array([s.decode('utf-8') for s in bunch.data])
    memo[param_hash]['matrix'] = text_matrix
    memo[param_hash]['features'] = []
    return text_matrix, bunch, []  # no features

  # raw occurences
  vectorizer = CountVectorizer(
    ngram_range=ngramRange,
    stop_words='english',
    min_df=minDF,
    vocabulary=None,
    binary=binary,
    token_pattern=r'(-?[Cc]\d+\b)|((?u)\b\w\w+\b)',  # enable neg capture
    analyzer=analyzer,
  )
  count_matrix = vectorizer.fit_transform(bunch.data)

  # save features
  features = []
  for feature in vectorizer.get_feature_names():
    features.append(feature)
  memo[param_hash]['features'] = features

  if pre_task == 'count':
    memo[param_hash]['matrix'] = count_matrix
    return count_matrix, bunch, features

  # tf-idf
  tf = TfidfTransformer()
  tfidf_matrix = tf.fit_transform(count_matrix)
  memo[param_hash]['matrix'] = tfidf_matrix
  return tfidf_matrix, bunch, features

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

def CrossVal(numFolds, classifier, matrix, bunch, pp_hash, clf_hash):
  # KFold
  kf_hash = hash_sum('%d%s%s' % (numFolds, pp_hash, clf_hash))
  if kf_hash in memo: return memo[kf_hash]['p'], memo[kf_hash]['r'], memo[kf_hash]['f1'], memo[kf_hash]['std'], memo[kf_hash]['mis']
  memo[kf_hash] = {}
  ps = []
  rs = []
  f1s = []
  folds = KFold(n_splits=numFolds)
  misses = []

  for train_indices, test_indices in folds.split(matrix):
    x_train = matrix[train_indices]
    y_train = bunch.target[train_indices]
    x_test = matrix[test_indices]
    y_test = bunch.target[test_indices]
    model = classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    if DEBUG: pdb.set_trace()
    misses += GetMisses(y_test, pred, bunch.filenames[test_indices])
    ps.append(precision_score(y_test, pred, pos_label=1))
    rs.append(recall_score(y_test, pred, pos_label=1))
    f1s.append(f1_score(y_test, pred, pos_label=1))
  misses = list(set(misses))
  misses.sort()
  p, r, f1, std = np.mean(ps), np.mean(rs), np.mean(f1s), np.std(np.array(f1s))
  memo[kf_hash]['p'] = p
  memo[kf_hash]['r'] = r
  memo[kf_hash]['f1'] = f1
  memo[kf_hash]['std'] = std
  memo[kf_hash]['mis'] = misses
  return p, r, f1, std, misses


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
  return p, r, f1, 0

def GetMisses(y_test, pred, names):
  # assumes bunch keeps it all in order (it does)
  misses = []

  for idx in range(len(y_test)):
    if not y_test[idx] == pred[idx]: misses.append(names[idx])
  return misses

def main():
  s_time = currentTime()
  global memo, gSParams
  g_size = 1
  sess_hash = hash_sum(str(gSParams))
  curr_sess = '%sf_bow_session_%s.pkl' % (dataDir, sess_hash)  # current session
  resume = 0
  for p in gSParams: g_size *= len(p)
  writeLog('%s: Generating call grid of size %d...' % (currentTime(), g_size))

  if os.path.exists(curr_sess):
    sess = loadPickle(curr_sess)
    results = sess['results']
    memo = sess['memo']
    resume = sess['last_idx']
    writeLog('Continuing session saved at %s from index %d' % (curr_sess, resume))

  else:
    results = gridSearchAOR(gSParams, doEval=False, funcList=[gSGenericRunner])
  writeLog('%s: Processing method calls %s to %s' % (currentTime(), results[resume], results[-1]))
  #pdb.set_trace()

  for idx in range(len(results)):
    if idx < resume: idx = resume

    try:
      writeLog('\n%s: Processing #%d of %d: %s' % (currentTime(), idx + 1, len(results), results[idx]))
      results[idx] = [results[idx], eval(results[idx])]

    except KeyboardInterrupt:
      writeLog('%s: Process INTerrupted by user. Saving progress...' % (currentTime()))
      sess = {'results': results, 'gsparams': gSParams, 'memo': memo, 'last_idx': idx}
      savePickle(sess, curr_sess)
      writeLog('%s: Successfully saved to %s' % (currentTime(), curr_sess))
      return

    except Exception as e:
      if DEBUG:
        results[idx] = 'Error in #%d: %s.\nSaving progress...' % (idx, repr(e))
        writeLog('%s: %s' % (currentTime(), results[idx]))
        sess = {'results': results, 'gsparams': gSParams, 'memo': memo, 'last_idx': idx}
        savePickle(sess, curr_sess)
        writeLog('%s: Successfully saved to %s' % (currentTime(), curr_sess))

      else:
        results[idx] = 'Exception in #%d: %s.' % (idx+1, repr(e))
        writeLog('%s: %s' % (currentTime(), results[idx]))
        raise
  results.append(gSParams)
  ex_num = getExpNum(dataDir + 'tracking.json')
  rf_name = '%sexp%s_anc_notes_GS_results' % (dataDir, ex_num)
  saveJson(results, rf_name + '.json')
  savePickle(memo, '%sexp%s_memo.pkl' % (dataDir, ex_num))
  if os.path.exists(curr_sess): os.remove(curr_sess)
  e_time = currentTime()
  fin_msg = 'Operation complete for experiment #%d. Started %s and ended %s.' % (ex_num, s_time, e_time)
  writeLog(fin_msg)
  slack_post(fin_msg, '@aphillips')
  commit_me(dataDir + 'tracking.json')

if __name__ == "__main__":
  main()
