#! /home/aphillips5/envs/nlpenv/bin/python3

# regular imports
from copy import deepcopy

from common import *


DEBUG = True
COMMA_SUB = ';'  # replace commas in CUI real names with this
usage = 'Usage: %s top {criteria} /path/to/exp_results' % (sys.argv[0])
#resolve_cuis = False
NA = 'N/A'


def get_fields_in_json(args):
    j_cont = loadJson(args[2])
    if not isinstance(j_cont, list): raise Exception('%s should be a list' % args[2])
    fields = []

    for item in j_cont:
        if not 'f1' in item[1]: continue
        item[1] = list(item[1])
        item[1].sort()
        fields = ', '.join(item[1])
        break
    if not fields: raise Exception('Couldn\'t find an F1 field. %s doesn\'t seem to be a valid results file' % args[2])
    print(fields)
    return

def get_top_results(critr, path, ext='json'):
    # 17-08-01
    cr_hash = hash_sum(critr)
    critr = str_to_dict(critr, '&', '=')
    print('Given criteria "%s" with hash "%s"' % (critr, cr_hash))
    memo_client = MemoryClient(sys.argv[0])
    print('Attempting load from memory server . . .')
    m_cont = memo_client(hash_sum(path))
    if not type(m_cont) in [list, dict]: m_cont = None
    if not m_cont: print('%s: Loading from %s . . .' % (currentTime(), path))
    j_cont = m_cont if m_cont else loadJson(path) if ext == 'json' else load_yaml(path) if ext == 'yaml' else None
    path = path.replace('.yaml', '.json')  # temp compat hack
    if not isinstance(j_cont, list): raise Exception('%s should be a list' % path)
    if not m_cont: memo_client(hash_sum(path), j_cont)
    top = {'f1': 0.0, 'recall': 0.0, 'precision': 0.0, 'accuracy': 0.0, 'auc': 0.0, 'acc': 0.0, 'spc':0.0, 'npv': 0.0}
    s_pat = 'UMLS_API_KEY='
    e_pat = '$'
    umls_key = re_findall('%s.+%s' % (s_pat, e_pat), loadText(os.environ['HOME'] + '/CREDS'), 0)[len(s_pat):] #(len(e_pat) * -1)]
    umls_clt = UMLSClient(umls_key, dataDir + 'umls_cache.json')
    umls_clt.gettgt()
    print('%s: Searching for best matching criteria...' % currentTime())
    optimize = critr['optimize'] if 'optimize' in critr and critr['optimize'] in 'precision|recall|f1|acc|auc|spc|npv' else 'f1'
    resolve_cuis = critr['resolve_cuis'] if 'resolve_cuis' in critr and critr['resolve_cuis'] in 'yes|no' else 'no'
    #pdb.set_trace()

    for idx in range(len(j_cont)):
        # seek max specified score
        targ = j_cont[idx][1]
        if not 'f1' in targ or targ['f1'] == None: continue
        skip = False

        for cr in critr:
            if skip: break

            if cr in targ:
                # is a score
                score_within_limits = False

                try:
                    if critr[cr][:2] in ['<=', '>=', '==']:
                        score_within_limits = eval('%s%s' % (targ[cr], critr[cr]))

                        if not score_within_limits:
                            skip = True
                            break

                except:
                    pass
            if not cr in targ['options']: continue  # if criteria outside run params
            t_cr = targ['options'][cr]
            if isinstance(t_cr, str) and re.match(critr[cr], t_cr): continue  # if regex match
            skip = True  # invalidate current result due to non-match
            break  # no need to go further with current
        if skip: continue  # if invalidated
        if targ.get(optimize, 0.0) <= top.get(optimize, 0.0): continue  # if not better than current
        top = deepcopy(targ)  # set new best
    top['criteria'] = critr
    ff_name = path_name_prefix('feats-%s_' % cr_hash, path).replace('.json', '.csv')
    mf_name = path_name_prefix('miscat-%s_' % cr_hash, path).replace('.json', '.txt')
    rf_name = path_name_prefix('top-res-%s_' % cr_hash, path)
    if not isinstance(top, list): top = [top]
    tmp_top = deepcopy(top)
    if 'features' in tmp_top: del tmp_top[0]['features']
    if 'mis' in tmp_top: del tmp_top[0]['mis']
    try:
        saveJson(tmp_top, rf_name)
    except:
        rf_name = rf_name.replace('.json', '.yaml')
        save_yaml(tmp_top, rf_name)
    print('Saved main results to file!')

    if 'features' in top[0] and top[0]['features']:
        # write features to a file
        print('%s: Writing features file...' % currentTime())
        #if DEBUG: pdb.set_trace()

        with open(ff_name, 'w') as fo:
            fo.write('cui/name,%s' % (','.join('fold_' + str(i) for i in range(len(top[0]['features'][0]) - 2))) + '\n')

            for feat in top[0]['features']:
                name = feat[0] + feat[1]  # assume one is always empty

                if umls_clt and resolve_cuis == 'yes' and (re.match('-?[Cc]\d{7,7}', name) or re.match('(neg)?[Cc]\d{7,7}', name)):
                    # found a cui to resolve
                    cui_name = name.lstrip('-').upper()
                    if cui_name.startswith('NEG'): cui_name = cui_name[3:]
                    if not len(cui_name) == 8: raise ValueError('CUI must be formatted C#######, got "%s"' % cui_name)
                    real_name = umls_clt.find_cui(cui_name).get('name', NA).replace(',', COMMA_SUB)
                    name = '%s (%s)' % (name, real_name)
                feat = '%s,%s\n' % (name, ', '.join(str(f) for f in feat[2:]))
                fo.write(feat)
            if umls_clt: umls_clt.save_cache()
        top[0]['features'] = ff_name

    if 'mis' in top[0]:
        # write misclassifications to a file
        print('%s: Writing misclassifications file...' % currentTime())

        with open(mf_name, 'w') as fo:
            fo.write('\n'.join(top[0]['mis']))
        top[0]['mis'] = mf_name
    try:
        saveJson(top, rf_name)

    except:
        rf_name = rf_name.replace('.json', '.yaml')
        save_yaml(top, rf_name)
    fin_msg = '\n%s: Top results for "%s" with criteria "%s" hash "%s":\n%s\nSaved to %s' % (currentTime(), path, critr, cr_hash, top, rf_name)
    writeLog(fin_msg)

    try:
        slack_post(fin_msg, '@aphillips')

    except OSError as e:
        fin_msg = '\n%s: Top results for "%s" with criteria "%s" hash "%s"\nsaved to %s' % (currentTime(), path, critr, cr_hash, rf_name)
        slack_post(fin_msg, '@aphillips')
    return top

def get_gs_params(path):
    results = None

    try:
        result = loadJson(path)

    except:
        result = load_yaml(path)

    for result in results:
        if 'methods' in result[0]: return result
    return 'Couldn\'t find grid search parameters'

def main(args):

    if len(args) == 1:
        print('No args given. Quitting...')
        return

    if args[1] == 'fields':
        if not os.path.exists(args[2]): args[2] = dataDir + args[2]
        if not os.path.exists(args[2]): raise Exception('%s doesn\'t exist' % args[2])
        if args[2].endswith('.json'): get_fields_in_json(args)
        return

    if args[1] == 'top':
        if not os.path.exists(args[3]): args[3] = dataDir + args[3]
        if not os.path.exists(args[3]): raise Exception('%s doesn\'t exist' % args[3])
        if args[3].endswith('.json'): get_top_results(args[2], args[3])
        if args[3].endswith('.yaml'): get_top_results(args[2], args[3], 'yaml')
        return

    if args[1].replace('-', '', 1).isdigit():
        pass

if __name__ == '__main__':
    try:
        main(sys.argv)
        commit_me(dataDir + 'tracking.json', 'export_results.py')

    except Exception as e:
        print('Exception: %s\n%s' % (repr(e), usage))
        pdb.post_mortem()
