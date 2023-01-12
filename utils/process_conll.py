'''
Creates a single file containing all of the sentences
provided by the SemEval task dataset for all languages
for the directory provided in the arguments
'''

import argparse
import json
import glob
import os
from os.path import join
import stanza
from bnlp import POS

bn2upos = { 'ALC': 'ADV', 'AMN': 'ADV', 'CCD': 'CCONJ', 'CCL': 'PART', 'CSB': 'SCONJ', 'CX': 'PART', 'CIN': 'INTJ', 'DAB': 'PRON', 'DRL': 'PRON', 'DWH': 'PRON', 
    'JJ': 'ADJ', 'JQ': 'DET', 'LC': 'VERB', 'LV': 'VERB', 'NC': 'NOUN', 'NP': 'PROPN', 'NST': 'NOUN', 'NV': 'NOUN', 'PP': 'ADP', 'PPR': 'PRON', 'PRF': 'PRON', 
    'PRC': 'PRON', 'PRL': 'PRON', 'PU': 'PUNCT', 'PWH': 'PRON', 'RDF': 'X', 'RDS': 'SYM', 'RDX': 'X', 'VAUX': 'AUX', 'VM': 'VERB'
    }


def main(args):
    data_path = join(args.path, f'*-{args.split}.conll')
    files = glob.glob(data_path)

    dataset = []
    for f in files:
        domain_ds = read_conllu(f)
        sents = [obj['tokens'] for obj in domain_ds]
        domain = domain_ds[0]['domain']
        tags = pos_tag_batch(sents, domain)
        for i,obj in enumerate(domain_ds):
            obj['pos_tags'] = tags[i]['pos_tags']
            obj['dep_tags'] = tags[i]['dep_tags']
        dataset.extend(domain_ds)

    ds_json = json.dumps(dataset, ensure_ascii=False)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    with open(join(args.out_path, f'{args.split}.json'), 'w', encoding='utf8') as outfile:
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
        'tokens': ['when', 'was', 'apple', 'founded'],
        'ner_tags': ['O', 'O', 'B-CORP', 'O']
        }
    """

    dataset = []
    with open(filename, encoding='utf-8') as f:
        sentence = []
        ner_tags = []
        obj = {}
        for row in f:
            row = row.strip()

            # This sentence is complete, add to list and initialize next one
            if len(row) == 0 and len(sentence) > 0:
                obj['tokens'] = sentence
                obj['ner_tags'] = ner_tags
                dataset.append(obj)
                sentence = []
                ner_tags = []
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
                ner_tags.append(row.split()[3])
                
    return dataset

def pos_tag_batch(sents, domain):
    '''
    Processes a batch of sentences for a given language (domain) and returns the POS tags
    
    Parameters
    ----------
    sents       A list of lists containing word-tokenized sentences in the same language
    domain      The language to be processed

    Returns a list of lists containing the POS tags for each sentence

    '''
    # Stanza doesn't support Bangla
    if domain != 'bn':
        nlp = stanza.Pipeline(lang=domain, processors='tokenize,pos,lemma,depparse',
                          verbose=True, use_gpu=True, tokenize_pretokenized=True)
        doc = nlp(sents)
        tags = [{'pos_tags': [word.upos for word in sent.words],
                'dep_tags': [word.deprel for word in sent.words]}
            for sent in doc.sentences]
    elif domain == 'bn':
        tags = []
        bn_pos = POS()
        model_path = "utils/bn_pos.pkl"
        for s in sents:
            doc = bn_pos.tag(model_path, s)
            tags.append({
                'pos_tags': [bn2upos[x[1]] for x in doc],
                'dep_tags': ['' for x in doc]})
    return tags

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str,
        help='The path to a directory containing .conll files to be processed into a single dataset',
        default='C:\\Users\\siena\\Dropbox\\Tuebingen\\Thesis\\MultiCoNER_2_train_dev\\train_dev')
    parser.add_argument('-o', '--out_path', type=str,
        help='The path to the directory in which to save the json file',
        default='.\data')
    parser.add_argument('-s', '--split', type=str,
        help='Either "train" or "dev" to indicate which files to process')

    args = parser.parse_args()

    assert args.split in ['train', 'dev']

    main(args)