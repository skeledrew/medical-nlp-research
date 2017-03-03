#! /NLPShare/nlpenv/bin/python3


from common import *


def catNotes(notes, noteTypes, destDir):
    ntype = noteTypes

    for mrn in notes:
        # access list of note entries for each patient
        data = notes[mrn]

        for note in data:
            # access each entry for a patient (assumed to be a list with final element being a diags list
            # diags list should have the misuse field last, auditscore 2nd to last

            try:
                
                if note[-3] in ntype:
                    # check the note type field

                    with open(destDir + note[-1][-1] + '/' + mrn + '.txt', 'a') as f:
                        f.write(note[-2])

            except Exception as e:
                print('Error: ' + str(e))
    return

if __name__ == '__main__':
    catNotes(pickleLoad(allNotesWithDiags), pickleLoad(ancNoteTypes), dataDir + 'anc_notes/')
    catNotes(pickleLoad(allNotesWithDiags), pickleLoad(labNoteTypes), dataDir + 'lab_notes/')
    catNotes(pickleLoad(allNotesWithDiags), pickleLoad(otherNoteTypes), dataDir + 'other_notes/')
