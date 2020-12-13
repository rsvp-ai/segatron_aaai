import os
import json
from tqdm import tqdm
import re
from multiprocessing import Pool

input_dir = 'out_txts/clean'
output_dir = 'docs'

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

def no_english(line):
    english_check = re.compile(r'.*[a-zA-Z]+.*')
    if re.match(english_check, line):
        return False
    else:
        return True
def is_split_line(line):
    words = line.strip().split()
    word_num = len(words)
    if word_num==0:
        return False
    if word_num<=10:
        if no_english(line):
            return True
        if 'chapter' in line.lower():
            return True
        if 'part' in line.lower():
            return True
        if words[0][0].isdigit():
            return True
    if word_num == 1 and words[0][0].isupper():
        return True
    return False
def remove_header(all_lines):
    top_1000 = all_lines[:1000]
    for index,line in enumerate(top_1000):
        if index>=995:
            return None
        if is_split_line(line):
            if is_split_line(top_1000[index+1]) or is_split_line(top_1000[index+2]) or is_split_line(top_1000[index+3]) or is_split_line(top_1000[index+4]):
                continue
            break
    return all_lines[index:]
def split_docs(lines):
    return None
def judge_newline(lines):
    sample_lines = [l for l in lines[len(lines)//4:len(lines)//2] if len(l.strip().split())>8]
    count_end_with_alphabet = sum([1 for l in sample_lines if l.strip()[-1].isalpha()])
    if count_end_with_alphabet/len(sample_lines)>0.3:
        return False
    else:
        return True
def doc_split(input_file):
    print(input_file)
    with open(input_file, 'r', encoding='UTF-8-sig') as ifile:
        lines = ifile.readlines()
        # lines = remove_header(book_str)
        # if not lines:
        #     lines = book_str
        parsed = {}
        parsed['text'] = ''
        unfinished = False
        for line in lines:
            # line = line.replace(u'\u3000', u' ').replace(u'\xa0', u' ')
            words = line.strip().strip('\t').split()
            if '.jpg' in line:
                continue
            # alpha_words = [w for w in words if w != "" and w[0].isalpha() ]
            if parsed['text'] != '':
                unfinished = parsed['text'][-2].isalpha() or parsed['text'][-2] == '-'
            if (len(words) == 0 or (
                    len(words) < 15 and (words[-1][-1].isalpha() or words[-1][-1].isdigit()))) and not unfinished:
                para_sep_line = True
            else:
                para_sep_line = False
            if para_sep_line:
                if parsed['text'] == '':
                    continue
                parsed['text'] = parsed['text'].strip() + '\n'
            elif len(words) > 0:
                parsed['text'] += line.strip() + ' '
            else:
                continue
        parsed['text'] = parsed['text'].replace('  ', ' ')
        return parsed

listOfFiles = getListOfFiles(input_dir)

p = Pool(96)
start_idx = 0
end_idx = 0
while start_idx < len(listOfFiles):
    end_idx = start_idx + 96 * 5
    results = p.map(doc_split,listOfFiles[start_idx: end_idx])
    with open(output_dir + str(end_idx), 'w') as ofile:
        for parsed in results:
            ofile.write(json.dumps(parsed) + '\n')
    start_idx = end_idx


# ab = []
# with open(output_file,'w') as ofile:
#     for input_file in tqdm_notebook(listOfFiles,total=len(listOfFiles)):
#         with open(input_file, 'r') as ifile:
#             book_str = ifile.readlines()
#             lines = remove_header(book_str)
#             if not lines:
#                 lines = book_str
#             try:
#                 newline_para = judge_newline(lines)
#             except:
#                 ab.append(input_file)
#                 continue
#             parsed = {}
#             paragraphs = []
#             if newline_para:
#                 # separated with \n
#                 for line in lines:
#                     words = line.strip().split()
#                     if '.jpg' in line:
#                         continue
#                     if len(words)<8 or no_english(line):
#                         if paragraphs==[] or len(' '.join(paragraphs))<100:
#                             paragraphs=[]
#                             continue
#                         else:
#                             parsed['text'] = '\n'.join(paragraphs)
#                             parsed['book'] = input_file
#                             if len(parsed['text'])>=500:
#                                 ofile.write(json.dumps(parsed) + '\n')
#                             paragraphs=[]
#                             parsed = {}
#                     else:
#                         paragraphs.append(line.strip())
#                         if len(paragraphs)>=6 and len(' '.join(paragraphs))>5000:
#                             parsed['text'] = '\n'.join(paragraphs)
#                             parsed['book'] = input_file
#                             if len(parsed['text'])>=500:
#                                 ofile.write(json.dumps(parsed) + '\n')
#                             paragraphs=[]
#                             parsed = {}
#                 else:
#                     stack = []
#                     for line in lines:
#                         words = line.strip().split()
#                         if '.jpg' in line:
#                             continue
#                         if len(words)<8 or no_english(line):
#                             if stack:
#                                 if stack[-1][-1].isalpha() and len(words)==1:
#                                     stack.append(line.strip())
#                                     continue
#                                 paragraphs.append(" ".join(stack).strip().replace('\n', ' '))
#                                 stack = []
#                                 if len(paragraphs)>=6 and len(' '.join(paragraphs))>5000:
#                                     parsed['text'] = '\n'.join(paragraphs)
#                                     parsed['book'] = input_file
#                                     if len(parsed['text'])>=500:
#                                         ofile.write(json.dumps(parsed) + '\n')
#                                     paragraphs=[]
#                                     parsed = {}
#                                     stack = []
#                             if paragraphs==[] or len(' '.join(paragraphs))<100:
#                                 paragraphs=[]
#                                 continue
#                             else:
#                                 parsed['text'] = '\n'.join(paragraphs)
#                                 parsed['book'] = input_file
#                                 if len(parsed['text'])>=500:
#                                     ofile.write(json.dumps(parsed) + '\n')
#                                 paragraphs=[]
#                                 parsed = {}
#                                 stack = []
#                         else:
#                             stack.append(line.strip())



