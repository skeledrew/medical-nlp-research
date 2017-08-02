#! /home/aphillips5/envs/nlpenv/bin/python3


from common import *
import csv


holder = {}
BAC_POS = 64

def mod(name, content):
    mrn = name.split('.')[0].lstrip('0')
    content = ''

    for row in holder['tfc']:

        #if not len(row[3]) == 7: raise Exception('Invalid MRN size. Fix needed')
        if not row[3] == mrn: continue
        #pdb.set_trace()

        if row[BAC_POS].lower() == 'no' or row[BAC_POS] == '0':
            content += ' BAC_NO'
            holder['mods'] += 1
            break
        if row[BAC_POS] == '': break

        if row[BAC_POS] == 'yes'.lower() or int(row[BAC_POS]) > 0:
            content += ' BAC_YES'
            holder['mods'] += 1
            break
    return content

def main(s_path, d_path):
    files = list(getFileList(s_path))
    ensureDirs(d_path)
    tf_csv = baseDir + 'Trauma_Final_20170614.csv'
    holder['tfc'] = [row for row in csv.reader(open(tf_csv), delimiter=',')]
    holder['mods'] = 0

    for name in files:
        name = name.split('/')[-1]

        with open(s_path + name) as sfo, open(d_path + name, 'w') as dfo:
            dfo.write(mod(name, sfo.read()))
    print('%d files modified' % holder['mods'])

if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2])
        commit_me(dataDir + 'tracking.json')

    except Exception as e:
        print('Exception: %s' % str(e))
        pdb.post_mortem()
