#! /home/aphillips5/envs/nlpenv/bin/python3


'''
Get Patients
- Extract patient data into individual files named by MRN
- Save patient data as JSON'd dict
- 17-06-14 - filter out unicode and non-printables
'''


from common import *
import string


patNotes = baseDir + 'alcohol_notes_01102017.txt'
#audits = dataDir + 'cutDiags_4_6-11_26_34-35_39-44.csv'


def loadNotes(noteFile):
    #
    print('Loading notes from ' + noteFile)
    notes = []
    cnt = 0

    with codecs.open(noteFile, 'r', encoding='utf-8', errors='ignore') as f:
        for row in f:
            if not '||' in row: continue
            notes.append(row.strip().split('||'))
            cnt += 1
    print('Loaded %d note records' % cnt)
    return notes

def saveNotes(noteList, destDir, id_col=0):
    # write out
    cnt = 0
    err = 0
    files = 0
    audNotes = {}
    ensureDirs(destDir)
    print('Saving notes as individual files')

    for note in noteList:
        cnt += 1

        try:
            name = destDir + note[id_col] + '.txt'

            with open(name, 'a') as f:
                f.write(''.join(ch for ch in '||'.join(note) if ch in string.printable) + '\n')

        except Exception as e:
            err += 1
            writeLog('%s: Error in %s: %s' % (currentTime(), sys.argv[0], str(e)))

        try:
            audNotes[note[id_col]].append(note)

        except KeyError:
            files += 1  # better to inc here
            audNotes[note[id_col]] = []
            audNotes[note[id_col]].append(note)

        except Exception as e:
            err += 1
            writeLog('Error at %s in %s for MRN %s: %s'(currentTime(), sys.argv[0], str(note[id_col]), str(e)))
    saveJson(audNotes, adeAllNotes)
    print('Total records: %d, total created files: %d, total errors: %d' % (cnt, files, err))

if __name__ == '__main__':
    notesFile = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else patNotes
    destDir = sys.argv[2] if len(sys.argv) > 2 else dataDir + 'note_files/'
    id_col = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    saveNotes(loadNotes(notesFile), destDir, id_col)
commit_me(dataDir + 'tracking.json', 'get_patients.py')
