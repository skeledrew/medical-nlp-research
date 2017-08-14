#! /home/aphillips5/envs/nlpenv/bin/python3


from common import *


oldDir = dataDir + 'anc_notes/'
newDir = dataDir + 'anc_notes_trim/'
substances = ['alcohol', 'beer', 'wine', 'liquor', 'scotch', 'bourbon', 'cognac', 'vodka', 'gin', 'jack daniels', 'brandy', 'hennessy', 'remy', 'alcoholic beverage', 'jagermeister']
posTriggers = ['etoh', 'disorient', 'syncope', 'impair', 'decrease', 'drink', 'deficit', 'thc', 'intoxicat', 'banana bag', 'lorazepam', 'ativan', ' b1', ' b6', 'thiamine', 'pyridoxine', 'multivitamin', 'bac ', 'bal ', 'dependence', 'heavy', 'admits', 'pint', 'shots', 'fifth', 'pack', 'binge drink']
genTriggers = ['fall', 'fell', 'dizz', 'nausea', 'vomit', 'lethargic', 'drowsy', 'shot', 'sluggish', 'male', 'groggy', 'cans?', '\b.{,3}\d.{,3}oz', 'day', 'week', 'daily', 'weekly', 'bottle', 'glass(es)?', 'gallon', ]
negTriggers = ['denies', 'none detected', 'sober']
triggers = substances + posTriggers + genTriggers + negTriggers


def grabTriggerSections(args):
    subDirs = ['yes/', 'no/']
    if len(args) > 1: global oldDir, newDir; oldDir = args[1]; newDir = args[2]
    ensureDirs(newDir + subDirs[0], newDir + subDirs[1])

    for subDir in subDirs:
        noteFiles = fileList(oldDir + subDir)

        for nfn in noteFiles:
            content = []
            meat = []

            with open('%s%s%s' % (oldDir, subDir, nfn)) as nf:

                for note in nf:
                    content.append(note.strip())
            content = makeStruct(content)

            for line in content:
                line = line.lower()  # lowercase every line

                try:

                    if line[-1] == ':' and len(line) > 2:
                        #print(line)
                        meat.append(line)
                        continue

                except:
                    # something's wrong with the line?
                    continue

                for lNum in range(5):
                    lNum = ' ' + str(lNum) + ' '
                    subst = False

                    for itm in substances:

                        if itm in line:
                            subst = True
                            break

                    if subst == True and lNum in line:
                        line = line.replace(lNum, ' NUMBER_LOW' + lNum)

                for hNum in range(5, 10):
                    hNum = ' ' + str(hNum) + ' '
                    subst = False

                    for itm in substances:

                        if itm in line:
                            subst = True
                            break

                    if subst == True and hNum in line:
                        line = line.replace(hNum, ' NUMBER_HIGH' + hNum)

                for trigger in triggers:
                    # search the words of interest

                    if re.search(trigger, line):
                        # add lines with the target word
                        # -08-12 - change to regex search
                        meat.append(line)
                        #print(line)
                        continue

            with open('%s%s%s' % (newDir, subDir, nfn), 'w') as nf:

                for line in meat:
                    nf.write(line + '\n')

def makeStruct(content):
    # takes a patient's notes and breaks into sections
    lines = []
    #print(content[-1])

    for note in content:
        # 1 note set per line
        #print('-' + str(note) + '-')
        #noteLst = re.split(':\s+', note)  # split on label colons
        noteLst = note.split(': ')

        for line in noteLst:
            label = line.split('  ')[-1]  # get the label (assume 1 or 0 spaces within label)
            #print(label)
            line = '  '.join(line.split('  ')[:-1])  # remove the label
            #lineLst = re.split('[\s\S+]+\.\s', line)  # try to get sentences and avoid title splits
            #lineLst = line.split('. ')  # splits titles; may affect results
            lineLst = re.sub(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", '\n', line, 0).split('\n')
            lines += lineLst
            lines += label.strip() + ':'
    return lines

if __name__ == '__main__':
    try:
        print('%s: Working...' % currentTime())
        grabTriggerSections(sys.argv)
        print('%s: Trimming operation complete.' % currentTime())

    except Exception as e:
        print('Exception: %s' % repr(e))
        pdb.post_mortem()
commit_me(dataDir + 'tracking.json', 'peel.py')
