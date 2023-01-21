# SemEval 2023 Task 2: MultiCoNER II
This is an implementation of multilingual complex named entity recognition as described in the SemEval 2023 Task 2: MultiCoNER II task description.

## Setup
This repository was written using Python 3.8.10 on Windows. It requires at least Python 3.7 (for the `evaluate` package)

When cloning this repository, first install all dependencies in your environment.

`pip install -r requirements.txt`

In your CLI, set the `PYTHONPATH` variable to the path of the repository.

## Preprocessing the corpus
To consolidate the corpus into one train and one dev set json, run the following code from the root of the repository. This will automatically save the output `train.json` or `dev.json` file to a `data` directory. This can also be specified using `-o`.

train.json: `python utils/process_conll.py -p <path-to-corpus-files> -s train`

dev.json:   `python utils/process_conll.py -p <path-to-corpus-files> -s dev`

This step also uses Stanza and bnlp-toolkit (for Bangla) to add POS and dependency relation tags (currently no UD tags for Bangla) to the files in addition to what is already found in the corpus, which is what causes this to take a little longer to run. It is recommended to run it using a GPU (default).

Note: For preprocessing test data, see the "Predictions" section.

## Models
Models for training can be found in the `train` directory.

### Base model
The `train_base.py` is a basic transformers implementation of the NER task. The purpose of it is to provide a baseline of how well a transformer model can perform without external/additional data.

### Multitask model
The `train_multitask.py` combines the task of NER tagging with POS tagging and dependency relation tags. It uses the same shared encoder for both tasks, while using separate heads. The goal is to see whether adding additional POS information and training simultaneously on both tasks improves results.

## Hyperparameter Tuning
There are jupyter notebooks for performing hyperparameter sweeps via `wandb`. First configure the sweep parameters in the file, and then run the notebook.

If you would prefer to avoid using jupyter, there are .py files as well. First configure the sweep parameters in the corresponding .yaml file. Note that if you are using a virtual environment, you will have to specify the location of the python executible to use. Otherwise, you can remove the `command` section. Then, in your CLI from the `hyperparameter-tuning` directory, run `wandb sweep --project <wandb project> <path to .yaml file>`. Make note of the sweep ID. Next run `wandb agent --count <num runs> <sweep ID>` to begin the sweep. Be sure to use the full ID path provided from the previous step.

## Predictions
Scripts for preditions can be found in the `predict` directory. By default, prediction files will be generated under `predict/predictions`.

The `predict_multitask.py` file can only be used with dev (labeled) data. It is made to be used with the same dataset that the training/tuning scripts use, produced by the `utils/process_conll.py` script.

The `predict_multitask_test.py` script makes predictions on a test set without any labels. To generate this test set, run the test .conll file through `utils/process_test_conll.py`. This script takes a file, rather than a directory like the other script, and outputs a JSON file to feed into the prediction script.