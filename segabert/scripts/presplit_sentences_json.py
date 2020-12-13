"""
Usage:
python scripts/presplit_sentences_json.py <original loose json file> <output loose json file>
"""

import sys
import json
import os
import nltk
from tqdm import tqdm
#nltk.download('punkt')


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles
mode = 'pst'
dirName = '/home/user/dataset/wikibook/wikipedia'
listOfFiles = getListOfFiles(dirName)
output_file = os.path.join(dirName, 'wikipedia_lines.json')
# dirName = '/home/user/data/data/bookcorpus_pos'
# listOfFiles = ["/home/user/data/data/bookcorpus/bookcorpus_lines.json"]
# output_file = os.path.join(dirName, 'bookcorpus_lines.json')
line_seperator = "\n"

#input_file = sys.argv[1]
with open(output_file, "w") as ofile:
    for input_file in tqdm(listOfFiles, total=len(listOfFiles)):
        with open(input_file, 'r') as ifile:
            for doc in tqdm(ifile.readlines()):
                parsed = json.loads(doc)
                sent_list = []
                para_indexes = []
                sent_indexes = []
                current_para = 0
                # parsed['text'] = parsed['text'].replace("\n","")
                for line in parsed['text'].split('\n'):
                    if line != '':
                        if mode=='pst':
                            sents = nltk.tokenize.sent_tokenize(line)
                            sent_list.extend(sents)
                            sent_indexes.extend([str(x) for x in range(len(sents))])
                            para_indexes.extend([str(current_para)] * len(sents))
                            current_para += 1

                parsed['text'] = line_seperator.join(sent_list)
                parsed['sent_pos'] = line_seperator.join(sent_indexes)
                parsed['para_pos'] = line_seperator.join(para_indexes)
                ofile.write(json.dumps(parsed) + '\n')

