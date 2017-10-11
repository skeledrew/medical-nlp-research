#! /home/aphillips5/envs/nlpenv/bin/python3

'''
Add the 
'''


import csv, sys
from os.path import exists

from common import *

### convention bridges
load_json = loadJson


def add_diags(args):
    pass

def main(args):

    if not len(args) > 5: return
    _, notes, notes_cr, diags, diags_cr, notes_with_diags = args
    notes_dict = load_json(notes) if exists(notes) else None
    if not notes_dict: raise ValueError('Cannot load notes from {}'.format(notes))
    diags = CSVWrapper(diags).make_dict(int(diags_cr)) if exists(diags) and diags_cr.isdigit() else None
    errs = cnt = 0
if __name__ == '__main__':
    main(sys.argv)
