'''
Creates a basic file containing all of the sentences
provided by the SemEval task dataset for a given language's
test file, which can then be loaded as a huggingface dataset
'''

import argparse
import json
import os
from os.path import join


def main(args):
    dataset = read_conllu(args.file)
    domain = dataset[0]['domain']

    ds_json = json.dumps(dataset, ensure_ascii=False)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    outpath = join(args.out_path, f'{domain}.test.json')
    print(f'Writing {len(dataset)} examples to {outpath}')
    with open(outpath, 'w', encoding='utf8') as outfile:
        outfile.write(ds_json)

def read_conllu(filename):
    """
    Read a (task-defined) CoNLL-U file, and return a list of objects representing each sentence.

    Parameters
    ----------
    filename    The name of the CoNLL-U file

    Returns a list of sentences in the following format:
    {
        'id': '0d88e010-c6e8-4409-9dec-a785e43eac16',
        'domain': 'en',
        'tokens': ['when', 'was', 'apple', 'founded']
        }
    """

    dataset = []
    with open(filename, encoding='utf-8') as f:
        sentence = []
        obj = {}
        for row in f:
            row = row.strip()

            # This sentence is complete, add to list and initialize next one
            if len(row) == 0 and len(sentence) > 0:
                obj['tokens'] = sentence
                dataset.append(obj)
                sentence = []
                obj = {}
            
            elif len(row) == 0:
              continue

            # If it is a comment, add the data to the object
            elif row[0] == '#' and len(row.split()) == 4:
                obj['id'] = row.split()[2]
                obj['domain'] = row.split()[3].split('=')[1]

            # If it is not a comment, add words to sentence
            else:
                sentence.append(row.split()[0])
                
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', type=str,
        help='The path to the .conll test file to be processed',
        default='C:\\Users\\siena\\Dropbox\\Tuebingen\\Thesis\\public_data\\DE-German\\de_test.conll')
    parser.add_argument('-o', '--out_path', type=str,
        help='The path to the directory in which to save the json file',
        default=join('.', 'data'))

    args = parser.parse_args()

    main(args)
