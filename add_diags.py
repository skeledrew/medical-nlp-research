#! /NLPShare/nlpenv/bin/python3

'''
Read cut diags and add to notes
'''

from common import *
import csv
from fuzzywuzzy import fuzz


def loadDiags(diags=''):
    # takes the JSON file path
    if not diags: diags = sys.argv[1]
    dDict = {}

    with open(diags) as df:
        dReader = csv.reader(df, delimiter=',')

        for row in dReader:
            if row[2] in dDict: raise Exception('Duplicate MRN %s detected in %s' % (row[2], diags))
            mrn = row.pop(2)
            dDict[mrn] = row
    print('Loaded %d records from %s' % (len(dDict), diags))
    return dDict

def addDiags(notes):
    noteDict = loadJson(notes) if isinstance(notes, str) and os.path.exists(notes) else notes if isinstance(notes, dict) else None
    if not noteDict: raise Exception('Cannot load notes')
    diags = loadDiags(dataDir + 'cutDiags_2-4_6-11_26_34-35_39-44.csv')
    errs = 0
    cnt = 0
    print('# MRNs in notes: %d, # MRNs in diags: %d' % (len(noteDict), len(diags)))
    #mrnPat = re.compile('[0-9]{5,8}')

    for mrn in noteDict:
        match = False
        cnt += 1

        if mrn in diags:
            # exact match; fastest
            noteDict[mrn].append(diags[mrn][2:])
            continue

        for diag in diags:
            # n^2 time search

            if len(diag) < 7 and re.match('0+%s' % diag, mrn):
                # 0s in front, assume len(mrn) == 7
                noteDict[mrn].append(diags[diag][2:])
                writeLog('%s: %s info: %s and %s matched via regex' % (currentTime(), sys.argv[0], mrn, diag))
                match = True
                break
            nameInNote = noteDict[mrn][0][1].lower()
            nameInDiag = '%s,%s'.lower() % (diags[diag][0], diags[diag][1])

            if fuzz.WRatio(mrn, diag) >= 95 and fuzz.WRatio(nameInNote, nameInDiag) >= 93:
                # last stab at a match via fuzzing
                noteDict[mrn].append(diags[diag][2:])
                writeLog('%s: Warning in %s: Possibly unreliable match between %s %s and %s %s' % (currentTime(), sys.argv[0], mrn, nameInNote, diag, nameInDiag))
                match = True
                break
        if match: continue
        writeLog('%s: Error in %s: Unknown MRN %s' % (currentTime(), sys.argv[0], mrn))
        errs += 1

    '''for mrn in diags:
        cnt += 1
        if len(mrn) == 6: mrn = '0' + mrn

        if not mrn in noteDict:
            writeLog('%s: Error in %s: Unknown MRN %s' % (currentTime(), sys.argv[0], mrn))
            errs += 1
            continue
        try:
            noteDict[mrn].append(diags[mrn])

        except KeyError:
            errs += 1
            writeLog('%s: Error in %s: MRN %s not found in notes' % (currentTime(), sys.argv[0], mrn))'''
    print('Total records processed: %d, errors: %d' % (cnt, errs))
    return noteDict

if __name__ == '__main__':
    dNotes = addDiags(allNotes)
    saveJson(dNotes, allNotesWithDiags)
