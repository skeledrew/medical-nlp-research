'''
Common
- For importing only
- Common operations and variables for interactive interpreter and scripts
- Recommended use: `from common import *`
'''


import pickle
import codecs
import sys
import re
import os
from random import shuffle
import math
import shutil
import json


STRING_TYPE = type('')
LIST_TYPE = type([])

baseDir = '/NLPShare/Alcohol/'
dataDir = baseDir + 'data/'

ancNoteTypes = dataDir + 'anc_note_types.lst'
labNoteTypes = dataDir + 'lab_note_types.lst'
otherNoteTypes = dataDir + 'other_note_types.lst'
allNotes = dataDir + 'all_notes.dct'
allDiags = dataDir + 'all_diags.dct'
allNotesWithDiags = dataDir + 'all_notes_with_diags.dct'

### For pickling operations

def pickleLoad(fName):
    obj = None
    
    with open(fName, 'rb') as fo:
        obj = pickle.load(fo)
    return obj

def pickleSave(obj, fName):

    with open(fName, 'wb') as fo:
        pickle.dump(obj, fo)
    return

### For command line operations

def readStdin():
    lines = []

    for line in sys.stdin.readlines():
        lines.append(line.strip())
    return lines

def writeStdout(lines, ofile=None):
    temp = None

    if not ofile == None:
        try:
            temp = sys.stdout
            sys.stdout = open(ofile, 'a')

        except Exception as e:
            print('Failed to use ' + ofile)

    for line in lines:
        sys.stdout.write(line)

    if not temp == None and not temp == sys.stdout:
        sys.stdout.close()
        sys.stdout = temp
    return

def writeStderr(lines):

    for line in lines:
        sys.stderr.write(line)
    return

def getCmdLine():
    return sys.argv

### For data search & manipulation

def pyCut(lines, delimiter=',', columns=['0'], errors='ignore'):
    '''
    Generator to cut lines based on column #s (1-based) and possibly rearrange order
    - Should work with strings and arrays
    - NB: may be broken by dynamically generated lists
    '''
    nline = None
    err = 0

    for line in lines:
        line = line.split(delimiter)

        try:

            for col in columns:

                if not ':' in col:
                    # simple base form
                    nline.append(line[int(col)])

                else:
                    splice = col.split(':')
                    nline.append(line[int(splice[0]) : int(splice[1])])
                    #raise Exception('Range splices not yet supported!')
            yield nline + '\n'  # should create a generator that returns individual lines
            
        except Exception as e:
            writeStderr(e.args)
            err += 1

def pyGrep(lines, pattern):
    pattern = re.compile(pattern)

    for line in lines:

        if not re.search(pattern, line) == None:
            yield line

def getFileList(path, recurse=False):
    
    for dirname, dirnames, filenames in os.walk(path):

        if not recurse:
            # TODO: move out of 'for'
            dirList = os.listdir(path)
            if not path[-1] == '/': path += '/'

            for fName in dirList:
                yield path + fName
            return
        # print path to all subdirectories first.
        #for subdirname in dirnames:
        #    print(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            print(os.path.join(dirname, filename))

def gridSearchAOR(p=None, construct='', results=[], doEval=False):
    # params is a list of dicts/lists of lists
    params = [{'methods': ['method1', 'method2']}, ['pos1arg1', 'pos1arg2'], ['pos2arg1', 'pos2arg2'],
              {'key1': ['a1-1', 'a1-2']}, {'key2': ['a2-1', 'a2-2']}] if p == None else p[:]
    #results = []

    if not params == []:
        # grab and process the first param
        param = params.pop(0)

        if type(param) == type({}):
            # process dictionary
            kName = ''
            for key in param:
                kName = key

            for item in param[kName]:
                result = None

                if kName == 'methods':
                    # processing the methods
                    result = gridSearchAOR(params, item + '(', results)  # start constructing method call

                else:
                    # processing named args

                    if type(item) == type('') and not item == 'False' and not item == 'True' and not item == 'None':
                        item = '\"%s\"' % item
                    result = gridSearchAOR(params, '%s %s=%s,' % (construct, kName, item), results)
                    #if result[-1] == ')' and not construct == '': return result

                if construct == '' and not result == []:
                    # back on top
                    #results.append(result)
                    pass

        elif type(param) == type([]):
            # process list, ie positional args

            for item in param:
                # processing positional args

                if type(item) == type('') and not item == 'False' and not item == 'True' and not item == 'None':
                    item = '\"%s\"' % item
                result = gridSearchAOR(params, '%s %s,' % (construct, item), results)

    else:
        # no more params to process
        result = construct[:-1] + ' )' # complete method call
        if not result in results: results.append(result)
        return result
    if not construct == '': return  # Only continue if we're at the top level
    if not doEval: return results

    for idx in range(len(results)):
        # evaluate them all
        print('Evaluating call #%d %s...' % (idx, results[idx]))

        try:
            results[idx] = [results[idx], eval(results[idx])]

        except Exception as e:
            print('Error in #%d, %s' % (idx, str(e.args)))
            results[idx] = [results[idx], str(e.args)]

    print('Grid search complete! Returning results.')
    return results

def getExpNum(tracker=''):
    # get and increment the experiement number on each call for autonaming
    tracking = pickleLoad(tracker)
    expNum = tracking['exp_num']
    tracking['exp_num'] += 1
    pickleSave(tracking, tracker)
    return expNum

def fileList(path, fullpath=False):
    nameList = os.listdir(path)

    if fullpath:

        for idx in range(len(nameList)):
            nameList[idx] = path + '/' + nameList[idx]
    return nameList

def splitDir(srcDir, destDir, percentOut, random=True, test=False):
    content = fileList(srcDir, True)
    numOut = len(content) - math.ceil(percentOut / 100 * len(content))  # take from end

    if random:
        shuffle(content)

    if test:
        print('Old dir: %s\n\nnew dir: %s' % (content[:numOut], content[numOut:]))

    else:

        for path in content[numOut:]:
            shutil.move(path, destDir)
    print('Moved %d of %d files to %s' % (len(content) - numOut, len(content), destDir))
    #return content[:numOut], content[numOut:]

def calcScores(tp=0, fp=0, fn=0, tn=0):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1

def loadJson(fName):
    obj = None

    with open(fName) as fo:
        obj = json.load(fo)
    return obj

def saveJson(obj, fName):

    with open(fName, 'w') as fo:
        json.dump(obj, fo)
    return

if __name__ == '__main__':
    print('This is a library module not meant to be run directly!')
