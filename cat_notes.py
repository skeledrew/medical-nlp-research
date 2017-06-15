#! /NLPShare/nlpenv/bin/python3


from common import *


def catNotes(notes, noteTypes, destDir):
    ntype = noteTypes.lower()
    cnt = 0
    err = 0
    print('Writing categorized notes to %s' % destDir)

    for mrn in notes:
        # access list of note entries for each patient
        data = notes[mrn]
        if not len(data[-1]) == 15: continue  # no audit data
        audit = data.pop()  # remove and return the audit data
        fDestDir = destDir + audit[-2].lower() + '/'
        ensureDirs(fDestDir)

        for note in data:
            # access each entry for a patient

            try:

                if note[-2].lower() in ntype:
                    # check the note type field
                    fName = fDestDir + mrn + '.txt'
                    if not os.path.exists(fName): cnt += 1

                    with open(fName, 'a') as f:
                        f.write(note[-1])

            except Exception as e:
                err += 1
                writeLog('Error at %s in %s: %s' % (currentTime(), sys.argv[0], str(e)), False)
    print('Total notes written: %d, errors: %d' % (cnt, err))
    return

if __name__ == '__main__':
    catNotes(loadJson(allNotesWithDiags), loadText(ancNoteTypes), dataDir + 'anc_notes/')
    catNotes(loadJson(allNotesWithDiags), loadText(labNoteTypes), dataDir + 'lab_notes/')
    catNotes(loadJson(allNotesWithDiags), loadText(otherNoteTypes), dataDir + 'other_notes/')
