# SemEval 2023 Task 2: MultiCoNER II
This is an implementation of multilingual complex named entity recognition as described in the SemEval 2023 Task 2: MultiCoNER II task description.

## Setup
This repository was written using Python 3.8.10 on Windows.

When cloning this repository, first install all dependencies in your environment.

`pip install -r requirements.txt`

## Preprocessing the corpus
To consolidate the corpus into one train and one dev set json, run the following code from the root of the repository. This will automatically save the output `train.json` or `dev.json` file to a `data` directory. This can also be specified using `-o`.

train.json: `python utils/process_conll.py -p <path-to-corpus-files> -s train`

dev.json:   `python utils/process_conll.py -p <path-to-corpus-files> -s dev`

This step also uses Stanza and bnlp-toolkit (for Bangla) to add POS tags to the files in addition to what is already found in the corpus, which is what causes this to take a little longer to run. It is recommended to run it using a GPU (default).

## Models
Models can be found in the `models` directory.

### Base model
The `base_model.py` is a basic transformers implementation of the NER task. The purpose of it is to provide a baseline of how well a transformer model can perform without external/additional data.