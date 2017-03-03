#! /NLPShare/nlpenv/bin/python3

from common import *

patNotes = baseDir + 'alcohol_notes_01102017.txt'

def loadNotes(noteFile):
    #
    notes =[]
    

    with codecs.open(noteFile, 'r', encoding='utf-8', errors='ignore') as f:
        for row in f:

            try:
                print(row.split('||')[-2])

            except:
                pass

if __name__ == '__main__':
    loadNotes(patNotes)
