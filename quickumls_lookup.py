#! /usr/bin/env python3


import pdb

from os.path import join, exists

from quickumls.client import get_quickumls_client
from common import *


usage = """
    Lookup medical terms in clinical notes with QuickUMLS

    :-s --src str none
        Input directory

    :-d --dest str none
        Output directory

    :-e --extract str CUI
        What to extract (currently only supports CUIs)

    :-u --subs str yes&no
        Subdirectory, separated by "&"

    :-t --store bool -
        Store processed data in the object

    :-ll --log-level str WARNING
        Log level (DEBUG|INFO|ERROR)

    :-lf --log-file str default
        File to log to
"""
log = None


class QuickUMLSLookup():

    def __init__(self, args):
        self._args = args
        self._matcher = get_quickumls_client()
        self._files = []
        self._analyses = []

    def lookup(self, text, extract=None):
        """Analyze given text and return analysis or specified part"""
        anal = self._matcher.match(text, best_match=True, ignore_syntax=False)
        if self._args.store: self._analyses.append(anal)

        if extract == 'CUI':
            cuis = ' '.join([seg[0]['cui'] for seg in anal])
            return cuis
        return anal

    def process(self, sub_dirs=None):
        """Processes the given directories"""
        sub_dirs = sub_dirs.split('&') if sub_dirs else ['yes', 'no']

        for sub in sub_dirs:
            src_dir = join(self._args.src, sub)
            if not exists(src_dir): raise OSError('Unable to find source "{}"'.format(src_dir))
            ensure_dirs(join(self._args.dest, sub))
            note_files = get_file_list(src_dir)

            for nfn in note_files:
                if self._args.store: self._files.append(join(sub, nfn))

                with open(join(self._args.src, sub, nfn)) as snf, open(join(self._args.dest, sub, nfn), 'w') as dnf:
                    #self.log.info('Processing')
                    dnf.write(self.lookup(snf.read(), 'CUI'))
        return self

def main():
    args = get_args(usage)
    if not args.src == 'none' and not args.dest == 'none':
        if not exists(args.log_file): args.log_file = dataDir + 'logs.txt'
        if not args.log_level.upper() in ['DEBUG', 'INFO', 'ERROR']: args.log_level = 'WARNING'
        log = get_log(__name__, level=args.log_level)
        log.info('Lookup started...')
        QuickUMLSLookup(args).process()
        log.info('Lookup complete!')
    return

if __name__ == '__main__':
    main()
