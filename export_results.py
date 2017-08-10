#! /home/aphillips5/envs/nlpenv/bin/python3

# regular imports

from common import *


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

def get_top_results(critr, path):
    # 17-08-01
    j_cont = loadJson(path)
    if not isinstance(j_cont, list): raise Exception('%s should be a list' % args[2])
    #critr = str_to_dict(critr, '&', '=')
    top = {'f1': 0.0}
    cr_hash = hash_sum(critr)

    for idx in range(len(j_cont)):
        targ = j_cont[idx][1]
        if not 'f1' in targ or targ['f1'] == None: continue
        ##check given criteria here
        if targ['f1'] <= top['f1']: continue
        top = targ

    if 'features' in top and top['features']:
        # write features to a file
        ff_name = path_name_prefix('feats-%s_' % cr_hash, path).replace('.json', '.txt')

        with open(ff_name, 'w') as fo:

            for feat in top['features']:
                fo.write(str(feat) + '\n')
        top['features'] = ff_name

    if 'mis' in top:
        # write misclassifications to a file
        mf_name = path_name_prefix('miscat-%s_' % cr_hash, path).replace('.json', '.txt')

        with open(mf_name, 'w') as fo:
            fo.write('\n'.join(top['mis']))
        top['mis'] = mf_name
    if not isinstance(top, list): top = [top]
    rf_name = path_name_prefix('top-res-%s' % cr_hash, path)
    saveJson(top, rf_name)
    writeLog('\n%s: Top results for "%s" with criteria "%s" hash "%s":\n%s\nSaved to %s' % (currentTime(), path, critr, cr_hash, top, rf_name))
    return top

def str_to_dict(s, main_sep, map_sep):
    # 17-08-01
    final = {}

    for item in s.split(main_sep):
        item = item.split(map_sep)
        final[item[0]] = item[1]
    return final

def main(args):
    print('Working...')

    if len(args) == 1:
        print('No args given. Terminating...')
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
        return

    if args[1].replace('-', '', 1).isdigit():
        pass

if __name__ == '__main__':
    try:
        main(sys.argv)
        commit_me(dataDir + 'tracking.json', 'export_results.py')

    except Exception as e:
        pdb.post_mortem()
