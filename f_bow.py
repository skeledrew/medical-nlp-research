#! /home/aphillips5/envs/nlpenv/bin/python3

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm, naive_bayes, linear_model, neighbors, ensemble, dummy
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix  # 17-09-25
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve  # 17-11-16

from common import *
import custom_clfs

ERROR_IGNORE = 'ValueError..eta0||TypeError..sequence||Failed to create||ValueError..Unsupported set||TypeError..A sparse matrix'
DEBUG = False
IGNORE = '~~IGNORE_THIS_PARAM~~'
memo = {}  # for memoization
clfMods = [
    svm, naive_bayes, linear_model, neighbors, custom_clfs, ensemble, dummy
]
DEFAULT_CONFIG = 'config.yaml'
#gSParams = config['gSParams']  # TODO: validate contents
custom_pp = ['text', 'bits']  # custom preprocessors


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
        alpha,
        lc_params,
        probability,
):
    result = Group()  # holds all the created objects, etc
    frame = currentframe()
    args, _, _, values = getargvalues(frame)
    result['options'] = allArgs = {arg: values[arg] for arg in args}
    preproc_hash = hash_sum('%s%s%d%s%s%s' %
                            (notesDirName, str(ngramRange), minDF, analyzer,
                             binary, preTask))
    #if DEBUG: pdb.set_trace()
    vals = PreProc(notesDirName, ngramRange, minDF, analyzer, binary, preTask,
                   preproc_hash)
    matrix, bunch, result['features'] = vals[0], vals[1], vals[2]
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
        'random_state': randState,  # make deterministic
        'alpha': alpha,
        'probability': probability,
    }
    classifier = MakeClf(clfName, hyParams, clfMods)
    result['classifier'] = re.sub('\n *', ' ', str(classifier))
    clf_hash = hash_sum(result['classifier'])
    if not clf_hash in memo: memo[clf_hash] = classifier
    sk_feats = False if preTask in custom_pp else True
    #pdb.set_trace()

    try:
        p = r = f1 = std = 0
        lc_params['train_set'] = notesDirName
        lc_params['mode'] = modSel
        lc_params['assoc_data'] = result
        #p, r, f1, std, mis, raw, others
        scores = CrossVal(
            numFolds, classifier, matrix, bunch, preproc_hash, clf_hash,
            result['features'], sk_feats
        ) if modSel == 'kf' else learning_curve(
            numFolds, classifier, matrix, bunch, preproc_hash, clf_hash,
            result['features'], sk_feats, lc_params
        )  #TTS(randState, classifier, matrix, bunch, preproc_hash, clf_hash)
        result['error'] = None
        if not isinstance(scores, Group): pdb.set_trace()
        [result(k, scores(k)) for k in scores]

    except (KeyError, IndexError) as e:
        print(repr(e))
        pdb.post_mortem()

    except Exception as e:
        ignore = False

        for err_pat in ERROR_IGNORE.split('||'):
            if re.search(err_pat, repr(e)): ignore = True

        if not ignore:
            print(repr(e))
            pdb.post_mortem()
        writeLog('%s: Error in classification: %s. Skipping...' %
                 (currentTime(), repr(e)[:80]))
        result = {
            'classifier': result['classifier'],
            'options': result['options'],
            #'others': []
        }
        result['error'] = e.args
        result['f1'] = result['precision'] = result['recall'] = result[
            'std'] = None
        #result['others'] = []
    #for other in result['others']:
    #    result[other] = result['others'][other]
    if result['f1'] and 'auc' in result:
        writeLog(
            '%s: Classifier %s \nwith options %s yielded: P = %s, R = %s, F1 = %s, Std = %s, AUC = %s, Accuracy = %s, Specificity = %s, NPV = %s'
            % (currentTime(), re.sub('\n *', ' ', str(result['classifier'])),
               str(result['options']), result['precision'], result['recall'],
               result['f1'], result['std'], result['auc'],
               result['acc'], result['spc'],
               result['npv']))
    return result

def PreProc(notesDirName, ngramRange, minDF, analyzer, binary, pre_task,
            param_hash):
    # 17-07-07 preprocessing with memoization for better speed and efficient memory use
    if param_hash in memo and not memo[param_hash]['matrix'] == None:
        return memo[param_hash]['matrix'], memo[param_hash]['bunch'], memo[
            param_hash]['features'], memo[param_hash]['pipe']
    memo[param_hash] = {}
    memo[param_hash]['matrix'], memo[param_hash]['bunch'], memo[param_hash][
        'features'], memo[param_hash]['pipe'] = None, None, [], []
    notesRoot = dataDir + notesDirName
    b_hash = hash_sum(notesRoot)
    bunch = memo[b_hash] if b_hash in memo else load_files(notesRoot)
    if not b_hash in memo: memo[b_hash] = bunch
    memo[param_hash]['bunch'] = bunch
    pipe = []  # hold transformer objects
    text_matrix = [s.decode('utf-8') for s in bunch.data]  # make str

    if pre_task == 'text':
        text_matrix = np.array(text_matrix)
        memo[param_hash]['matrix'] = bunch.data = text_matrix
        memo[param_hash]['pipe'] = pipe
        return text_matrix, bunch, [], pipe  # no features

    if pre_task == 'bits':
        bit_vec = custom_clfs.BitVectorizor(ngram_range=ngramRange)
        bits_matrix = np.array(bit_vec.fit_transform(text_matrix))
        memo[param_hash]['matrix'] = bunch.data = bits_matrix
        memo[param_hash]['pipe'] = pipe
        return bits_matrix, bunch, bit_vec._ent_list, pipe

    # raw occurences
    vectorizer = CountVectorizer(
        ngram_range=ngramRange,
        stop_words='english',
        min_df=minDF,
        vocabulary=None,
        binary=binary,
        token_pattern=
        r'(-?[Cc]\d+\b)|((?u)\b\w\w+\b)|(\b[a-zA-Z0-9_]{1,}\b)',  # enable neg capture, underscores
        analyzer=analyzer,
    )
    pipe.append(('vect', vectorizer))
    count_matrix = vectorizer.fit_transform(bunch.data)

    # save features
    features = []
    for feature in vectorizer.get_feature_names():
        features.append(feature)
    memo[param_hash]['features'] = features

    if pre_task == 'count':
        memo[param_hash]['matrix'] = bunch.data = count_matrix
        memo[param_hash]['pipe'] = pipe
        return count_matrix, bunch, features, pipe

    # tf-idf
    tf = TfidfTransformer()
    pipe.append(('tfidf', tf))
    tfidf_matrix = tf.fit_transform(count_matrix)
    memo[param_hash]['matrix'] = bunch.data = tfidf_matrix
    memo[param_hash]['pipe'] = pipe
    return tfidf_matrix, bunch, features, pipe

def MakeClf(clf_name, hyparams, clf_mods):
    # create classifier and add relevant hyperparameters
    clf = None
    mod = None
    classifier = None

    for m in clf_mods:
        # get the module containing the classifier
        # NB: doesn't handle possible duplicate class names in diff mods
        if not clf_name in m.__dict__: continue
        mod = m
        break
    if not mod:
        raise Exception(
            'Unable to find "%s" in any of the given modules.' % clf_name)
    clf = getattr(mod, clf_name)
    clf_str = str(clf())
    params = ', '.join([
        '%s=%s' %
        (p, hyparams[p]
         if not isinstance(hyparams[p], str) else hyparams[p].__repr__())
        for p in hyparams if p + '=' in clf_str and not hyparams[p] == IGNORE
    ])  # make parameter string

    try:
        classifier = eval('clf(%s)' % (params))

    except Exception as e:
        raise Exception('Failed to create %s with params %s; %s' % (clf_name,
                                                                    params,
                                                                    repr(e)))
    return classifier

def CrossVal(numFolds,
             classifier,
             matrix,
             bunch,
             pp_hash,
             clf_hash,
             feats,
             sk_feats=True,
             **rest_kw):
    # KFold
    kf_hash = hash_sum('%d%s%s' % (numFolds, pp_hash, clf_hash))
    if kf_hash in memo and memo[kf_hash]:
        result = Group(**memo[kf_hash])
        return result
        #return memo[kf_hash]['p'], memo[kf_hash]['r'], memo[kf_hash][
        #    'f1'], memo[kf_hash]['std'], memo[kf_hash]['mis'], memo[kf_hash][
        #        'raw']
    memo[kf_hash] = {}
    ps = []
    rs = []
    f1s = []
    raw_results = []  # holds tn, fp, fn, tp
    other_results = {}  # throw in everything else being calculated
    accs = []
    aucs = []
    spcs = []
    npvs = []
    rocs = []
    final_result = Group()
    folds = KFold(n_splits=numFolds)
    misses = []
    wghts_read = False
    #pdb.set_trace()

    for idx in range(len(feats)):
        if not sk_feats: break
        feats[idx] = [feats[idx][0], feats[idx][1]]
    f_idx = 0

    for train_indices, test_indices in folds.split(matrix):
        x_train = matrix[train_indices]
        y_train = bunch.target[train_indices]
        x_test = matrix[test_indices]
        y_test = bunch.target[test_indices]
        model = classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)
        pred_p = classifier.predict_proba(x_test) if hasattr(
            classifier, 'predict_proba') else None
        #if hasattr(classifier, 'coef_'):
        #    [
        #        feats[idx].append(classifier.coef_[0][idx])
        #        for idx in range(len(feats))
        #    ]
        misses += GetMisses(y_test, pred, bunch.filenames[test_indices])
        ps.append(precision_score(y_test, pred, pos_label=1))
        rs.append(recall_score(y_test, pred, pos_label=1))
        f1s.append(f1_score(y_test, pred, pos_label=1))
        raw = confusion_matrix(y_test, pred)
        raw = dict(enumerate(raw))  # harden against missing values
        raw[0] = dict(enumerate(raw.get(0, [0, 0])))
        raw[1] = dict(enumerate(raw.get(1, [0, 0])))
        raw = {
            'tn': int(raw[0].get(0, 0)),
            'fp': int(raw[0].get(1, 0)),
            'fn': int(raw[1].get(0, 0)),
            'tp': int(raw[1].get(1, 0))
        }
        raw_results.append(raw)
        accs.append(accuracy_score(y_test, pred))
        roc = list(
            roc_curve(y_test,
                      np.asarray([e[0] for e in pred_p])
                      if not type(pred_p) == type(None)
                      and pred_p.shape[1] == 2 else [0.0] * len(y_test)))
        roc[0] = [float(e) for e in roc[0]]
        roc[1] = [float(e) for e in roc[1]]
        roc[2] = [float(e) for e in roc[2]]
        rocs.append(roc)
        if not type(pred_p) == type(None) and not pred_p.shape == y_test.shape:
            y_test = np.asarray(make_one_hot(y_test))
        aucs.append(
            roc_auc_score(y_test, pred_p if not type(pred_p) == type(None)
                          and pred_p.shape == y_test.shape else
                          [0.0] * len(y_test)))
        #if not type(pred_p) == type(None): pdb.set_trace()
        spcs.append(raw['tn'] / ((raw['tn'] + raw['fp']) or 1))
        npvs.append(raw['tn'] / ((raw['tn'] + raw['fn']) or 1))
    misses = list(set(misses))
    misses.sort()
    p, r, f1, std, acc, auc, spc, npv = float(np.mean(ps)), float(
        np.mean(rs)), float(np.mean(f1s)), float(np.std(np.array(f1s))), float(
            np.mean(accs)), float(np.mean(aucs)), float(np.mean(spcs)), float(
                np.mean(npvs))
    raw_means = {
        key:
        sum(map(lambda result: result[key], raw_results)) / len(raw_results)
        for key in ['tn', 'fp', 'fn', 'tp']
    }
    raw_results.append(raw_means)
    rocs.append(average_roc_folds(rocs))
    memo[kf_hash]['precision'] = final_result['precision'] = p
    memo[kf_hash]['recall'] = final_result['recall']= r
    memo[kf_hash]['f1'] = final_result['f1']= f1
    memo[kf_hash]['std'] = final_result['std']= std
    memo[kf_hash]['mis'] = final_result['mis']= misses
    memo[kf_hash]['raw'] = final_result['raw']= raw_results
    memo[kf_hash]['acc'] = other_results['acc'] = acc
    memo[kf_hash]['auc'] = other_results['auc'] = auc
    memo[kf_hash]['spc'] = other_results['spc'] = spc
    memo[kf_hash]['npv'] = other_results['npv'] = npv
    memo[kf_hash]['rocs'] = other_results['rocs'] = rocs
    [final_result(r, other_results[r]) for r in other_results]
    if not isinstance(final_result, Group): pdb.set_trace()
    return final_result #p, r, f1, std, misses, raw_results, other_results

def TTS(randState, classifier, tfidf_matrix, bunch, pp_hash, clf_hash):
    # train-test split
    tts_hash = hash_sum('%d%s%s' % (randState, pp_hash, clf_hash))
    if tts_hash in memo:
        raise Exception('Arg combo already processed. Skipping...')
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

def main(args):
    #pdb.set_trace()
    state = Group()
    state('args', get_args())
    if state.args.eval: return test_eval(state)
    s_time = currentTime()
    cfg_file = state.args.config_file or DEFAULT_CONFIG
    writeLog('Loading config from "%s"' % cfg_file)
    config = load_yaml(cfg_file)
    state('config', config)
    gSParams = config.get('gSParams')
    global memo
    g_size = 1
    sess_hash = hash_sum(str(gSParams))
    curr_sess = '%sf_bow_session_%s.pkl' % (dataDir,
                                            sess_hash)  # current session
    resume = 0
    for p in gSParams:
        g_size *= len(p)
    writeLog('%s: Generating call grid of size %d...' % (currentTime(),
                                                         g_size))
    ### setup distributed crunching
    crunch_servers = config['crunch_servers']
    crunch_client = CrunchClient(procs=1.5)
    crunch_client.disabled = True
    setattr(crunch_client, 'gSGenericRunner', gSGenericRunner)
    save_progress = True
    save_err = 0

    if state.args.learning_curve:
        # set stage
        save_progress = False
        if not 'lc' in gSParams[9]: gSParams[9] = ['lc']

    for server in crunch_servers:
        res = crunch_client.add_server(server['host'], server['port'],
                                       server['name'], server['user'])
        writeLog('%s: %s' % (currentTime(), res))

    if os.path.exists(curr_sess):
        # continue from a saved session
        sess = loadPickle(curr_sess)
        results = sess['results']
        memo = sess['memo']
        resume = sess['last_idx']
        writeLog('Continuing session saved at %s from index %d' % (curr_sess,
                                                                   resume))

    else:
        results = gridSearchAOR(
            gSParams, doEval=False, funcList=[gSGenericRunner])
    if not results:
        raise ValueError('Grid failed. Params are: {}'.format(str(gSParams)))
    writeLog('%s: Processing method calls %s to %s' %
             (currentTime(), results[resume], results[-1]))
    #pdb.set_trace()

    for idx in range(len(results)):
        if idx < resume: idx = resume

        try:
            writeLog('\n%s: Processing #%d of %d: %s' %
                     (currentTime(), idx + 1, len(results), results[idx]))
            #if idx == 24: pdb.set_trace()  # for debugging tuple issue
            results[idx] = [results[idx], eval(results[idx]).to_dict()]
            #func = globals()[results[idx].split('(')[0]]
            #args = list(eval('(' + '('.join(results[idx].split('(')[1:]).strip(' ')))
            #crunch_client.add_task(func, args)

        except KeyboardInterrupt:
            writeLog('%s: Process INTerrupted by user. Saving progress...' %
                     (currentTime()))
            sess = {
                'results': results,
                'gsparams': gSParams,
                'memo': memo,
                'last_idx': idx
            }
            savePickle(sess, curr_sess)
            writeLog('%s: Successfully saved to %s' % (currentTime(),
                                                       curr_sess))
            return

        except Exception as e:
            ignore = False

            for err_pat in ERROR_IGNORE.split('||'):
                if re.search(err_pat, repr(e)): ignore = True

            if DEBUG and not ignore:
                results[idx] = 'Error in #%d: %s.\nSaving progress...' % (
                    idx + 1, repr(e))
                writeLog('%s: %s' % (currentTime(), results[idx]))
                sess = {
                    'results': results,
                    'gsparams': gSParams,
                    'memo': memo,
                    'last_idx': idx
                }
                savePickle(sess, curr_sess)
                writeLog('%s: Successfully saved to %s' % (currentTime(),
                                                           curr_sess))
                pdb.post_mortem()

            else:
                results[idx] = 'Exception in #%d: %s.' % (idx + 1, repr(e))
                writeLog('%s: %s' % (currentTime(), results[idx]))
                #raise
    crunch_client.wait()
    writeLog('%s: Crunching complete. Wrapping up...' % (currentTime()))
    #results = crunch_client.get_results()
    results.append(gSParams)
    ex_num = getExpNum(dataDir + 'tracking.json')
    rf_name = '%sexp%s_anc_notes_GS_results' % (dataDir, ex_num)
    try:
        if save_progress: saveJson(results, rf_name + '.json')
    except:
        save_err += 1
        save_yaml(results, rf_name + '.yaml')
    if save_progress: savePickle(memo, '%sexp%s_memo.pkl' % (dataDir, ex_num))
    if os.path.exists(curr_sess): os.remove(curr_sess)
    e_time = currentTime()
    save_err_msg = '' if not save_err else ' An error occurred during JSON save so YAML used instead.'
    fin_msg = 'Operation complete for experiment #%d. Started %s and ended %s. %s' % (
        ex_num, s_time, e_time, save_err_msg)
    writeLog(fin_msg)
    slack_post(fin_msg, '@aphillips')

def test_eval(state, **rest_kw):
    # takes results file/dict, result index, test dir
    args = state.args
    #if not os.path.exists(args.clfs_file):
    #    raise Exception('Invalid result file: %s' % args.clfs_file)
    results = None
    writeLog('%s: Loading results file...' % currentTime())
    if os.path.exists(args.clfs_file):
        results = loadJson(
            args[0]) if args[0].endswith('.json') else load_yaml(
                args.clfs_file) if args.clfs_file.endswith('.yaml') else args.clfs_file
    if isinstance(results, dict): results = [results]
    if not isinstance(results, list):
        raise ValueError('Invalid result format; must be a list.')
    save_progress = False if results[args.result_index][1]['options']['modSel'] in ['lc'
                                                                    ] else True
    #pdb.set_trace()
    if args.result_index < 0 or args.result_index > len(results) - 1:
        raise ValueError(
            'Invalid index; must be 0 or a positive integer less than %d' %
            len(results))
    result = results[args.result_index][1]
    if not os.path.exists(args.test_dir):
        raise Exception('Invalid path for test set')
    writeLog('%s: Args validated: %s' % (currentTime(), str(args)))
    test_set = args.test_dir.rstrip('/').split('/')[-1]
    params = result['options']
    hyparams = str_to_dict(
        re.split('\( *', result['classifier'])[-1][:-1], ', *', '=',
        True)  # get from clf string
    hyparams = {
        key: (val[1:-1] if val[0] in ['"', "'"] else float(val)
              if '.' in val else int(val) if val.isdigit() else True
              if val == 'True' else False if val == 'False' else None
              if val == 'None' else float(val)
              if val[0] in ['-', '+'] and '.' in val else int(val)
              if val[0] in ['-', '+'] and val[1:].isdigit() else val)
        for key, val in zip(hyparams.keys(), hyparams.values())
    }  # zap extra quotes & convert non-strings
    hyparams = {}

    for ccp in params:
        # fix bug caused by camelCase hyperparam names
        scp = cc_to_sc(ccp)
        hyparams[scp] = params[ccp]
    hyparams['random_state'] = 1
    classifier = MakeClf(params['clfName'], hyparams, clfMods)
    _, train_bunch, feats, pipe = PreProc(
        params['notesDirName'], params['ngramRange'], params['minDF'],
        params['analyzer'], params['binary'], params['preTask'], 'train_eval')
    #_, test_bunch, _, _ = PreProc(test_set, params['ngramRange'], params['minDF'], params['analyzer'], params['binary'], params['preTask'], 'test_eval')
    if 'clf' in pipe[-1]: pipe.pop(-1)  # remove old classifier
    pipe.append(('clf', classifier))
    clf_pipe = Pipeline(pipe)

    for idx in range(len(feats)):
        # make features holder into a list of lists
        if isinstance(feats[idx], list) and not len(feats[idx]) == 2: continue
        #writeLog('Detected a double feature: "%s" and "%s"' % (feats[idx][0], feats[idx][1]))
        feats[idx] = [feats[idx][0] + feats[idx][1]]
    x_train = train_bunch.data
    y_train = train_bunch.target
    test_bunch = preproc_test(test_set, pipe)
    x_test = test_bunch.data
    y_test = test_bunch.target
    writeLog('%s: Evaluating on test...' % currentTime())
    model = classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    if hasattr(classifier, 'coef_'):
        [
            feats[idx].append(classifier.coef_[0][idx])
            for idx in range(len(feats))
        ]
    misses = GetMisses(y_test, pred, test_bunch.filenames)
    misses = list(set(misses))
    p = precision_score(y_test, pred, pos_label=1)
    r = recall_score(y_test, pred, pos_label=1)
    f1 = f1_score(y_test, pred, pos_label=1)
    pred_p = auc = None

    try:
        pred_p = classifier.predict_proba(x_test) if hasattr(
            classifier, 'predict_proba') else None
        if not type(pred_p) == type(None) and not pred_p.shape == y_test.shape:
            y_test = np.asarray(make_one_hot(y_test))
        auc = roc_auc_score(y_test,
                            pred_p) if not type(pred_p) == None else None
    except Exception as e:
        writeLog('%s: Error while evaluating on test: %s' % (currentTime(),
                                                             repr(e)))
    mf_name = path_name_prefix('miscat-test_', args[0].replace(
        '.json', '.txt')) if save_progress else None
    if save_progress: saveText('\n'.join(misses), mf_name)
    ff_name = path_name_prefix('feats-test_',
                               args[0]) if save_progress else None
    if save_progress:
        saveText('\n'.join(', '.join(str(f)) for f in feats), ff_name)
    classifier = re.sub('\n *', ' ', str(clf_pipe.steps[-1][-1]))
    writeLog(
        '%s: Classifier %s \nwith options "%s..." on test set %s yielded: P = %s, R = %s, F1 = %s, AUC = %s'
        % (currentTime(), classifier, str(params)[:200], test_set, p, r, f1,
           auc))
    rf_name = path_name_prefix('test-res_', args[0]) if save_progress else None
    result = {
        'classifier': classifier,
        'options': params,
        'test_set': test_set,
        'P': p,
        'R': r,
        'F1': f1
    }
    if save_progress:

        try:
            saveJson(result, rf_name)

        except:
            writeLog('%s: JSON save failed. Falling back to YAML...' %
                     (currentTime()))
            save_yaml(result, rf_name.replace('.json', '.yaml'))
    return result

def test_it(args, kw):
    print('Test ran fine...')
    return 'Success!'

def learning_curve(*args):
    # 17-11-09
    pdb.set_trace()
    lc_p = args[8]
    args = list(args)  # to allow later modification
    train_result = lc_p['assoc_data']
    test_path = get_test_path('{}{}'.format(dataDir, lc_p['train_set']))
    lcf_name = path_name_prefix('learn-curve_{}-{}_'.format(
        lc_p['least'], lc_p['step']), test_path) + '.csv'
    curve_values = []
    whole_matrix = args[2].toarray()  # make iteratable
    whole_bunch = args[3]
    data_len = len(whole_bunch.target)
    classes = {}

    for idx, lbl in enumerate(whole_bunch.target):
        # separate classes
        if not lbl in classes:
            classes[lbl] = {'y': [], 'X': [], 'filenames': []}
        classes[lbl]['X'].append(whole_matrix[idx])
        classes[lbl]['y'].append(whole_bunch.target[idx])
        classes[lbl]['filenames'].append(whole_bunch.filenames[idx])

    for t_size in range(lc_p['least'], lc_p['most'], lc_p['step']):
        # get F1 at each step
        if t_size > 100: break
        t_size = t_size / 100  # %age
        samps = {}
        sub_bunch = Group()
        setattr(sub_bunch, 'target_names', whole_bunch.target_names)
        setattr(sub_bunch, 'DESCR', whole_bunch.DESCR)
        setattr(sub_bunch, 'target', [])
        setattr(sub_bunch, 'filenames', [])
        setattr(sub_bunch, 'data', None)
        sub_matrix = []
        random.seed(train_result['options']['randState'])

        for lbl in classes:
            # get portions
            classes_len = len(classes[lbl]['y'])
            samp_idxs = sorted(
                random.sample(range(classes_len), int(classes_len * t_size)))
            if not lbl in samps: samps[lbl] = {}

            for idx in samp_idxs:
                sub_bunch.target.append(classes[lbl]['y'][idx])
                sub_bunch.filenames.append(classes[lbl]['filenames'][idx])
                sub_matrix.append(classes[lbl]['X'][idx])
        args[2] = np.asarray(sub_matrix)
        sub_bunch.target = np.asarray(sub_bunch.target)
        sub_bunch.filenames = np.asarray(sub_bunch.filenames)
        args[3] = sub_bunch

        try:
            train_result = Group()
            train_result(lc_p['assoc_data'])
            p, r, f1, std, mis, raw = CrossVal(*args[:-1])
            train_result['precision'] = p
            train_result['recall'] = r
            train_result['f1'] = f1
            train_result['std'] = std
            train_result['mis'] = mis
            train_result['raw'] = raw
            train_result['error'] = None
            test_result = test_eval([train_result, 0, test_path])
            curve_values.append([t_size, test_result['F1']])

        except Exception as e:
            if 'BdbQuit' in repr(e): raise e
            print('Something broke: {}. Skipping...'.format(repr(e)))
            pass
    saveText('\n'.join(','.join(str(e) for e in v) for v in curve_values),
             lcf_name)
    #pdb.set_trace()
    return 0.0, 0.0, 0.0, 0.0, [], [], {}

def get_test_path(train_path):
    # 17-11-09
    # TODO: include 'train' in naming scheme to facilitate test name discovery
    if not os.path.exists(train_path):
        raise OSError('Train dataset "{}" does not exist'.format(train_path))
    path_parts = list(os.path.split(train_path))
    test_path = ''
    old_name = path_parts[-1]

    if 'train' in path_parts[-1]:
        # handle future name format
        path_parts[-1] = path_parts[-1].replace('train', 'test', 1)
        test_path = os.path.join(*path_parts)
        if os.path.exists(test_path): return test_path
        path_parts[-1] = old_name
    no_notes_match = re.match('([a-z]+_)', path_parts[-1])
    with_notes_pat = '\w+_notes_\w+'

    if no_notes_match and not re.search(with_notes_pat, path_parts[-1]):
        # verbose notes removed
        frag = no_notes_match.group(1)
        path_parts[-1] = path_parts[-1].replace(frag, '{}test_'.format(frag),
                                                1)
        test_path = os.path.join(*path_parts)
        if os.path.exists(test_path): return test_path
        path_parts[-1] = old_name

    if re.search(with_notes_pat, path_parts[-1]):
        # contains verbose notes
        path_parts[-1] = path_parts[-1].replace('_notes_', '_notes_test_', 1)
        test_path = os.path.join(*path_parts)
        if os.path.exists(test_path): return test_path
        test_path = ''
        path_parts[-1] = old_name
    raise OSError('Unable to find test set path from "{}"'.format(train_path))

def score(**kwargs):
    pass

def average_roc_folds(rocs):
    # 18-02-05 - find the mean of the points in each field of k folds of ROCs
    roc_ave = [None, None, None]
    if not isinstance(rocs, list) or False in [isinstance(f, list) for f in rocs]: return roc_ave
    #pdb.set_trace()
    err = None

    try:
        for fold in range(len(rocs)):

            for field in range(3):
                # fpr, tpr or thresh
                f_size = len(rocs[fold][field])  # field size, ie num points
                if not roc_ave[field]: roc_ave[field] = [0.0] * f_size

                for point in range(f_size):
                    roc_ave[field][point] = (
                        roc_ave[field][point] * fold + rocs[fold][field][point]
                    ) / (fold + 1)
    except Exception as e:
        print(repr(e))
        #pdb.set_trace()
        err = e
    return roc_ave

def preproc_test(test_set, pipe):
    # apply preprocess transformations to test data
    bunch = load_files(dataDir + test_set)
    matrix = bunch.data  #[s.decode('utf-8') for s in bunch.data]

    for trans in pipe:
        # apply each transformation
        if 'clf' in trans: break
        if not hasattr(trans[1], 'transform'): continue
        matrix = trans[1].transform(matrix)
    bunch.data = matrix
    return bunch

def get_args():
    from argparse import ArgumentParser as AP
    p = AP(description='Train and evaluate different model configurations')
    p.add_argument('--config-file', help='Config file path', type=str, default=DEFAULT_CONFIG)
    p.add_argument('--learning-curve', help='Flag to generate a learning curve', action='store_true')
    p.add_argument('--eval', help='Evaluate a model on a test set', action='store_true')
    p.add_argument('--test-dir', help='Test set path', type=str, default='')
    p.add_argument('--clfs-file', help='Result file with classifiers for eval', default='')
    p.add_argument('--result-index', help='Index of classifier in result file', type=int, default=0)
    p.add_argument('-mp', '--multiprocess', help='Use crunch service', action='store_true')
    args = p.parse_args()
    return args


if __name__ == "__main__":
    try:
        main(sys.argv)

    except Exception as e:
        print('Exception: %s' % (repr(e)))
        pdb.post_mortem()
commit_me(dataDir + 'tracking.json', 'f_bow.py')
