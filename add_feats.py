#! /home/aphillips5/envs/nlpenv/bin/python3

'''
17-08-02 - Add features from Trauma CSV to a data dir
  - TODO: accept mod_func object for more flexibility. done
    - add multiple features at once. done
'''


import csv

from common import *
from peel import substances as subs


usage = 'Usage: %s /path/to/src/dir/ /path/to/dest/dir/ mod1+mod2+...+modN' % (sys.argv[0])
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
        if bac_val > 130: content += ' CRITICAL_BAC'
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

def week_cons_add(content, row):
    # calculate and add a mean alcohol consumption (https://en.wikipedia.org/wiki/Blood_alcohol_content)
    # https://www.drugabuse.gov/sites/default/files/files/AUDIT.pdf
    freqs = {
            'day[^s]': 7,
            'daily': 7,
            'night': 7,
            'weekday': 5,
            'weekend': 2,
            'week[^s]': 1,
            'weekly': 1,
            'wk': 1,
            'month': 0.25,
            'monthly': 0.25,
    }
    types = {  # alcohol conc in %
            'beer': 5,
            'malt': 7,
            'wine': 12,
            'hard': 40,
    }
    subs = {  # beverage names as may appear in corpus
            'drink': 'beer',
            'wine': 'wine',
            'milwaukee best ice': 'beer',
            'liquor': 'hard',
    }
    sizes = {  # drink volume in oz
            'can': 12,
            'drink': 12,
            #'cup?': 8.5,  # malt qty
            'glass': 5,
            'bottle': 25,
            'shot': 1.5,
            'pint': 16,
            'fifth': 25,
            'large can': 22,
            '6.{,3}pack': 72,
            '12.{,3}pack': 144,
    }
    use_ref_lines = []
    sizes_re = r'|'.join('(%s)' % sz for sz in sizes)
    freqs_re = r'|'.join('(%s)' % fq for fq in freqs)
    subs_re = r'|'.join('(%s)' % sb for sb in subs)

    for line in content:
        # get lines with relevant usage references
        ref_ctr = len(re.findall(sizes_re, line) + re.findall(freqs_re, line) + re.findall(subs_re, line))
        if ref_ctr < 2: continue  # not enough use references in line
        ## <check for multiple refs>
        #line = [line]
        [use_ref_lines.extend(line) for line in split_multiple(line, ref_ctr)]
    drinks = []
    if use_ref_lines: pdb.set_trace()

    for line in use_ref_lines:
        # calculate drink amounts and frequencies
        quan = get_quan(line, sizes_re, subs_re, sizes, subs)  # get consumption in oz
        conc = types[subs[re.find(subs_re, line)]]  # ~ % of alcohol
        freq = get_freq(line, freqs_re)  # times per week
        drink = quan * (conc / 100) / 0.6  # oz * %age / 0.6
        drinks.append([drink, freq[0], freq[1]])
    #std_drinks = sum(drinks)  # num std drinks per week
    gender = 'm' if row[5] == 0 else 'f'
    bod_wat = 0.58 if gender == 'm' else 0.49  # body water constant m v f
    wt_def = 60  # give an 'idealized' weight as default
    wt  = int(row[33]) if row[33].isdigit() else 0
    wt_unit = row[34]
    weight = wt if 'kg' in wt_unit else wt * 0.45 if 'lb' in wt_unit else wt_def
    metabol = 0.017 if gender == 'f' else 0.015
    drink_perd = 2  # throw an average
    tot_cons = []

    for drink in drinks:
        # weekly drinking estimate
        est_bac = ((0.806 * drink[0] * 1.2) / (bod_wat * weight) - (metabol * drink_perd)) * 10 * 1000  # final g/dL -> mg/dL
        tot_cons.sppend(est_bac * drink[1] * drink[2])  # in a week
    tot_cons = sum(tot_cons)
    b_size = 50
    buck = tot_cons / b_size
    content += ' W_CONS_%s_%s W_CONS_VAL_%s' % (b_size, buck, tot_cons)
    return content

def split_multiple(line, ref_ctr):
        if ref_ctr < 4: return [line]
        return [line]

def get_quan(line, szre, sbre, sizes, subs):
    amt = re.find('\d+((\-|/|\.)\d+)?.{,3}((%s)|(%s))' % (szre, sbre), line)   # find range or fraction
    if amt: amt = re.find('\d+((\-|/|\.)\d+)?', amt)  # extract
    if '-' in amt: amt = (int(amt.split('-')[1]) + int(amt.split('-')[0])) / 2
    if '/' in amt: amt = float(amt.split('/')[0]) / float(amt.split('/')[1])
    if isinstance(amt, str) and '.' in amt: amt = float(amt)
    size = sizes[re.find(szre, line)] if re.search(szre, line) else 1
    if size: amt *= size
    #bevr = subs[re.find(sbre, line)]
    return amt

def get_freq(line, fqre):
    amt = re.find('\d+(\-\d)?.{,3}(\w+.{,3}){,5}(%s)' % fqre, line)
    if amt: amt = re.find('\d+(\-\d)?', amt)
    if '-' in amt: amt = (int(amt.split('-')[1]) + int(amt.split('-')[0])) / 2
    if not amt: amt = 2.5  # guess if nothing found
    f_unit = re.find(fqre, line)
    return amt, f_unit

def lower_zap(content, row):
        pdb.set_trace()
        content = re.sub('\b.*[^A-Z]+.*\b', '', content)  # zap all w/o uppers
        content = re.sub('\s{3,}', ' ', content)  # zap extra spaces
        return content

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
    mod_funcs = ['bac_yn_add', 'gender_add', 'race_add', 'week_cons_add', 'lower_zap']
    if not isinstance(mod_func, str): mod_funcs.append(mod_func)
    holder['tfc'] = [row for row in csv.reader(open(tf_csv), delimiter=',')]
    holder['cnt'] = 0
    holder['mod_funcs'] = mod_funcs

    for name in files:
        name = name.split('/')[-1]

        with open(s_path + name) as sfo, open(d_path + name, 'w') as dfo:
            funcs = mod_func.split('+')
            content = sfo.read()

            for func in funcs:
                content = mod(name, content, func)
            dfo.write(content)
    print('%d files modified' % holder['cnt'])

if __name__ == '__main__':
    try:
        if len(sys.argv) == 4 and os.path.exists(sys.argv[1]) and re.match('^[a-z_\+]+$', sys.argv[3]):
            # basic usage validation
            main(sys.argv[1], sys.argv[2], sys.argv[3])

        else:
            print(usage)
        commit_me(dataDir + 'tracking.json')

    except Exception as e:
        print('Exception: %s\n%s' % (str(e), usage))
        pdb.post_mortem()
