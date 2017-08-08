#! /home/aphillips5/envs/nlpenv/bin/python3

'''
17-08-02 - Add features from Trauma CSV to a data dir
  - TODO: accept mod_func object for more flexibility
    - add multiple features at once
'''

from common import *
import csv


holder = {}
BAC_C = 64
BAC_VAL = 63


def bac_yn_add(content, row):
    # add BAC y/n to content

        if row[BAC_C].lower() == 'no':
            content += ' BAC_NO'
            holder['cnt'] += 1
        #if row[BAC_POS] == '': break

        if row[BAC_C].lower() == 'yes':
            content += ' BAC_YES'
            holder['cnt'] += 1
        return content

def bac_yn_only(content, row):
    # replace content with BAC y/n

        if row[BAC_C].lower() == 'no':
            content = 'BAC_NO'
        #if row[BAC_POS] == '': break

        if row[BAC_C].lower() == 'yes':
            content = 'BAC_YES'
        return content

def bac_val_add(content, row):
    # add BAC value buckets to content
    b_size = 50  # bucket size
    b_size = int(b_size)
    if b_size <= 0: raise Exception('Bucket size must be greater than zero.')

    if row[BAC_VAL].isdigit():
        # only modifiy if it's an unsigned integer
        bac_val = int(row[BAC_VAL])
        buck = bac_val / b_size
        content += ' BAC_%d_%d BAC_VAL_%s' % (b_size, buck, bac_val)
        if bac_val > 175: content += ' CRITICAL_BAC'
    return content

def gender_add(content, row):
    # converts and adds gender value
    GEN_COL = 5

    if row[GEN_COL] == 0:
        content += ' GENDER_MALE'

    if row[GEN_COL] == 1:
        content += ' GENDER_FEMALE'
    return content

def race_add(content, row):
    RAC_COL = 6

    if row[RAC_COL].isdigit():
        content += ' RACE_%s' % row[RAC_COL]

    else:
        content += ' UNKNOWN_RACE'
    return content

def bac_all_add(content, row):
        #pdb.set_trace()
        return bac_val_add(bac_yn_add(content, row), row)

def mod(name, content, mod_func):
    mrn = name.split('.')[0].lstrip('0')

    for row in holder['tfc']:
        if not row[3] == mrn: continue  # get to the correct record
        if not mod_func in holder['mod_funcs']: raise Exception('Invalid modifier function %s given' % mod_func)
        #pdb.set_trace()
        if isinstance(mod_func, str): mod_func = mod.__globals__[mod_func]
        content = mod_func(content, row)
        holder['cnt'] += 1
    return content

def main(s_path, d_path, mod_func):
    files = list(getFileList(s_path))
    ensureDirs(d_path)
    tf_csv = baseDir + 'Trauma_Final_20170614.csv'
    mod_funcs = ['bac_yn_add', 'bac_yn_only', 'bac_all_add', 'gender_add', 'race_add']
    if not isinstance(mod_func, str): mod_funcs.append(mod_func)
    holder['tfc'] = [row for row in csv.reader(open(tf_csv), delimiter=',')]
    holder['cnt'] = 0
    holder['mod_funcs'] = mod_funcs

    for name in files:
        name = name.split('/')[-1]

        with open(s_path + name) as sfo, open(d_path + name, 'w') as dfo:
            funcs = mod_func.split('&')
            content = sfo.read()

            for func in funcs:
                content = mod(name, content, func)
            dfo.write(content)
    print('%d files modified' % holder['cnt'])

if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
        commit_me(dataDir + 'tracking.json')

    except Exception as e:
        print('Exception: %s' % str(e))
        pdb.post_mortem()
