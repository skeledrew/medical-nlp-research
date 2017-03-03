#! /NLPShare/nlpenv/bin/python3


'''
Get Patients
- Extract patient data into individual files named by MRN
- Pickle patient data as array object
'''


from common import *


patNotes = baseDir + 'alcohol_notes_01102017.txt'

def loadNotes(noteFile):
    #
    print('Loading notes from ' + noteFile)
    notes = []

    with codecs.open(noteFile, 'r', encoding='utf-8', errors='ignore') as f:
        for row in f:
            notes.append(row.split('||'))
    return notes

def saveNotes(noteList):
    # write out
    cnt = 0
    err = 0
    print('Saving notes as individual files')

    for note in noteList:
        cnt += 1

        try:
            name = dataDir + 'note_files/' + note[0] + '.txt'

            with open(name, 'a') as f:
                f.write('||'.join(note))

        except Exception as e:
            err += 1
    print('Total records: ' + str(cnt) + ', total errors: ' + str(err))

if __name__ == '__main__':
    saveNotes(loadNotes(patNotes))
