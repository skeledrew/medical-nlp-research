#! /home/aphillips5/envs/nlpenv/bin/python3

'''
17-08-02 - Add features from Trauma CSV to a data dir
  - TODO: accept mod_func object for more flexibility. done
    - add multiple features at once. done
'''


import csv

from common import *
from peel import substances as subs


DEBUG = True
mod_funcs = ['bac_yn_add', 'bac_val_add', 'gender_add', 'race_add', 'week_cons_add', 'lower_zap', 'merge', 'unique', 'bac_note_add']
usage = 'Usage: %s /path/to/src/dir/ /path/to/dest/dir/ mod1+mod2+...+modN\nMods are: %s' % (sys.argv[0], str(mod_funcs))
holder = {}
BAC_C = 64
BAC_VAL = 63


def bac_yn_add(content, row):
    # add BAC y/n to content

        if row[BAC_C].lower() == 'no':
            content += ' BAC_NO'
        #if row[BAC_POS] == '': break

        if row[BAC_C].lower() == 'yes':
            content += ' BAC_YES'
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
            'week ?days?': 5,
            'weekend': 2,
            'week[^sed]': 1,
            'weekly': 1,
            'wk': 1,
            'month': 0.25,
            'monthly': 0.25,
            'None': 0,
    }
    types = {  # alcohol conc in %
            'beer': 5,
            'malt': 7,
            'wine': 12,
            'hard': 40,
    }
    subs = {  # beverage names as may appear in corpus
            'drink': 'beer',
            'beer': 'beer',
            'wine': 'wine',
            'milwaukee best ice': 'beer',
            'liquor': 'hard',
            'hennessy': 'hard',
            'remy': 'hard',
    }
    sizes = {  # drink volume in oz
            #'oz': 1,
            '[^a-z]cans?[^a-z]': 12,
            #'drink': 12,
            #'cup?': 8.5,  # malt qty
            '[^a-z]glass(es)?': 5,
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
    if not re.search(subs_re, content) or not re.search(freqs_re, content): return content

    for line in list(set(content.split('\n'))):
        # get lines with relevant usage references
        ref_ctr = len(re_findall(sizes_re, line) + re_findall(freqs_re, line) + re_findall(subs_re, line))
        if ref_ctr < 2: continue  # not enough use references in line
        ## <check for multiple refs>
        #line = [line]
        [use_ref_lines.append(line) for line in split_multiple(line, ref_ctr)]
    drinks = []
    #if use_ref_lines: pdb.set_trace()
    if not use_ref_lines: return content

    for line in use_ref_lines:
        # calculate drink amounts and frequencies
        quan = get_quan(line, sizes_re, subs_re, sizes, subs)  # get consumption in oz
        conc = types[subs[re_findall(subs_re, line, 0)]] if re.search(subs_re, line) else 0  # ~ % of alcohol
        freq = get_freq(line, freqs_re)  # times per week
        drink = int(quan * (conc / 100) / 0.6)  # oz * %age / 0.6
        drinks.append([drink, int(freq[0]), freq[1]])
    tmp = set(['|'.join(str(i) for i in d) for d in drinks])  # remove dups
    drinks = [[int(i) if i.isdigit() else i for i in d.split('|')] for d in tmp]  # split out
    #std_drinks = sum(drinks)  # num std drinks per week
    gender = 'm' if row[5] == 0 else 'f'
    bod_wat = 0.58 if gender == 'm' else 0.49  # body water constant m v f
    wt_def = 60  # give an 'idealized' weight as default
    wt  = int(row[33]) if row[33].isdigit() else 0
    wt_unit = row[34]
    weight = wt if 'kg' in wt_unit else wt * 0.45 if 'lb' in wt_unit else wt_def
    if weight < 30 and re.search('(kg)|(lb)', wt_unit): weight = wt_def
    metabol = 0.017 if gender == 'f' else 0.015
    drink_perd = 2  # throw an average
    tot_cons = []

    for drink in drinks:
        # weekly drinking estimate
        if drink[0] <= 0 or drink[1] <= 0 or drink[2] == 'None': continue
        est_bac = ((0.806 * drink[0] * 1.2) / (bod_wat * weight) - (metabol * drink_perd)) * 10 * 100  # final g/dL -> mg/dL?
        drink[2] = re_index(drink[2], freqs)
        if not type(drink[2]) in [int, float]: drink[2] = 0
        tot_cons.append(est_bac * drink[1] * drink[2])  # in a week
    tot_cons = int(sum(tot_cons))
    b_size = 250
    buck = int(tot_cons / b_size)
    content += ' W_CONS_%s_%s W_CONS_VAL_%s' % (b_size, buck, tot_cons)
    return content

def split_multiple(line, ref_ctr):
        # detect and try to split multiple drinking instances
        if ref_ctr < 4: return [line]
        lines = re.split('and|,| {3,}', line)  # simple split
        return lines

def get_quan(line, szre, sbre, sizes, subs):
    # quantity in oz if possible
    amt = re_findall('a|(\d+((\-|/|\.)\d+)?).{,3}((%s)|(%s))' % (szre, sbre), line, 0)   # find range or fraction
    if not amt: return 0
    amt = re_findall('a|(\d+((\-|/|\.)\d+)?)', amt, 0)  # extract
    if '-' in amt: amt = (int(amt.split('-')[1]) + int(amt.split('-')[0])) / 2
    if isinstance(amt, str) and '/' in amt: amt = float(amt.split('/')[0]) / float(amt.split('/')[1])
    if amt == 'a': amt = 1
    if isinstance(amt, str): amt = float(amt) if  '.' in amt else int(amt)
    size = re_index(re_findall(szre, line, 0), sizes) if re.search(szre, line) else 1
    if size: amt *= size
    #bevr = subs[re_findall(sbre, line)
    return amt

def get_freq(line, fqre):
    amt = re_findall('the|(\d+(\-\d)?).{,3}(\w+.{,3}){,5}(%s)' % fqre, line, 0)
    if not amt: return 0, None
    amt = re_findall('the|(\d+(\-\d)?)', amt, 0)
    if '-' in amt: amt = (int(amt.split('-')[1]) + int(amt.split('-')[0])) / 2
    if amt == 'the': amt = 1
    if not amt: amt = 2.5  # guess if nothing found
    if isinstance(amt, str): amt = float(amt)
    f_unit = re_findall(fqre, line, 0)
    return amt, f_unit

def lower_zap(content, row):
        #pdb.set_trace()
        #content = re.sub('\b.*[^A-Z]+.*\b', '', content)  # zap all w/o uppers
        words = content.split(' ')

        for idx in range(len(words)):

            if not re.search('[A-Z]', words[idx]) or re.search('[a-z]', words[idx]):
                # no upper
                words[idx] = ''
        content = ' '.join(words)
        content = re.sub('\s{3,}', ' ', content)  # zap extra spaces
        return content

def merge(content, row):
    # dummy bypass with just space pad
    return ' %s ' % content

def unique(content, row):
    lines = content.split('\n')
    uniq_lines = []

    for line in lines:
        line = line.strip()
        if line in uniq_lines: continue
        uniq_lines.append(line)
    return '\n'.join(uniq_lines)

def bac_note_add(content, row):
    # add y/n and value if parsable; should prob do unique first
    lines = content.split('\n')
    bac_unk = ' NOTES_BAC_UNKNOWN'
    bac_yes = ' NOTES_BAC_YES'
    bac_no = ' NOTES_BAC_NO'
    bac_val = ' NOTES_BAC_VAL_'
    bac_crit = ' NOTES_BAC_CRITICAL'
    bac_clash = ' NOTES_BAC_PARSE_ISSUE_OR_NOTES_CONTRADICTION'
    status = bac_unk
    value = -1
    pats = [
            r'etoh {0,5}\d+',
            r'\+ {0,3}etoh',
            r'blood {0,5}alcohol .{0,15}\d+',
            r'\d .{0,10}blood {0,5}alcohol',
            r'elevated {0,5}blood {0,5}alcohol',
            r'blood {0,5}alcohol {0,5}elevated',
            r'[^a-z](negative|positive) {0,5}(bac|bal)[^a-z]',
    ]
    pos_pats = r'(\+|elevated|positive)'
    neg_pats = r'(\-|negative)'
    annot = status

    for line in lines:
        break_out = False

        for pat in pats:
            match = re.search(pat, line)
            if not match: continue
            match_text = match.group(0)
            if re.search(neg_pats, match_text): status = bac_no
            if re.search(pos_pats, match_text): status = bac_yes
            match = re.search('\d+', line)
            if not match: break_out = True; break  # we're done here
            match_num = int(match.group(0))
            if status == bac_unk: status = bac_no if match_num == 0 else bac_yes
            annot = status
            annot += '%s%d' % (bac_val, match_num)
            if match_num > 130: annot += bac_crit
            if (status == bac_yes and match_num == 0) or (status == bac_no and match_num > 0): annot += bac_clash
            break  # no need to check other patterns
        if break_out: break
    content += annot
    return content


def mod(name, content, mod_func):
    mrn = name.split('.')[0].lstrip('0')

    for row in holder['tfc']:
        if not row[3] == mrn: continue  # get to the correct record
        if not mod_func in holder['mod_funcs']: raise Exception('Invalid modifier function %s given' % mod_func)
        #pdb.set_trace()
        if isinstance(mod_func, str): mod_func = mod.__globals__[mod_func]
        content = mod_func(content, row)
    return content

def main(s_path, d_path, mod_func):
    files = list(getFileList(s_path))
    ensureDirs(d_path)
    tf_csv = baseDir + 'Trauma_Final_20170614.csv'
    global mod_funcs
    if not isinstance(mod_func, str): mod_funcs.append(mod_func)
    holder['tfc'] = [row for row in csv.reader(open(tf_csv), delimiter=',')]
    holder['cnt'] = 0
    holder['mod_funcs'] = mod_funcs
    writeLog('%s: Applying new feature(s) "%s" to "%s" from source "%s"' % (currentTime(), mod_func.replace('+', ', '), d_path, s_path))

    for name in files:
        name = name.split('/')[-1]
        mode = 'a' if 'merge' in mod_func else 'w'

        with open(s_path + name) as sfo, open(d_path + name, mode) as dfo:
            funcs = mod_func.split('+')
            content = sfo.read()

            for func in funcs:

                try:
                    content = mod(name, content, func)

                except Exception as e:
                    writeLog('%s: Exception in add_feats.py: %s' % (currentTime(), repr(e)))
                    if DEBUG: pdb.post_mortem()
                    continue
            dfo.write(content)
        holder['cnt'] += 1
    writeLog('%s: Feature application complete; %d files modified, new files in "%s"' % (currentTime(), holder['cnt'], d_path))

if __name__ == '__main__':
    try:
        if len(sys.argv) == 4 and os.path.exists(sys.argv[1]) and re.match('^[a-z_\+]+$', sys.argv[3]):
            # basic usage validation
            main(sys.argv[1], sys.argv[2], sys.argv[3])

        else:
            print(usage)
        commit_me(dataDir + 'tracking.json')

    except Exception as e:
        print('Exception: %s\n%s' % (repr(e), usage))
        pdb.post_mortem()
