import numpy as np
import transformers
from datasets import load_dataset
import wandb
from os.path import join
import os
import gc
import torch

import evaluate

from models import MULTI_LABEL2ID, MULTI_ID2LABEL
from models import MultitaskModel, NLPDataCollator, MultitaskTrainer

wandb.login()

data_dir = join(os.getcwd(), 'data')

# Set to None if training multilingual
lang = 'de'

model_name = 'xlm-roberta-base'

#############################################
#           Function Definitions            #
#############################################

def convert(obj, classes):
  obj['ner_tags_numeric'] = [classes[t] for t in obj['ner_tags']]
  obj['pos_tags_numeric'] = [classes[t] for t in obj['pos_tags']]
  obj['dep_tags_numeric'] = [classes[t] for t in obj['dep_tags']]
  return obj

def align_labels_with_tokens(labels, word_ids, task_name):
    # This function expects O to be 0 and all NER labels to follow in alternating B-I order
    # For tokens inside a word but not at the beginning, we replace the B- with I- (since the token does not begin the entity)
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special tokens get a label of -100
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            if task_name == 'ner':
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
            new_labels.append(label)

    return new_labels

def convert_to_ner_features(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, is_split_into_words=True,
        padding='max_length'
    )
    all_labels = examples['ner_tags_numeric']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids, 'ner'))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def convert_to_pos_features(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, is_split_into_words=True,
        padding='max_length'
    )
    all_labels = examples['pos_tags_numeric']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids, 'pos'))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def convert_to_dep_features(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, is_split_into_words=True,
        padding='max_length'
    )
    all_labels = examples['dep_tags_numeric']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids, 'dep'))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[MULTI_ID2LABEL[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [MULTI_ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

#############################################
#                Model Setup                #
#############################################

# Load the train and dev dataset files into a HuggingFace dataset
raw_datasets = load_dataset('json', data_files={
    'train': join(data_dir, "train.json"), 
    'validation': join(data_dir, "dev.json")
    },
    chunksize=40<<20,
    ignore_verifications=True)
if lang:
    raw_datasets = raw_datasets.filter(lambda example: example["domain"] == lang)
else:
    lang = 'multi'
raw_datasets = raw_datasets.map(convert, fn_kwargs={'classes': MULTI_LABEL2ID})

dataset_dict = {
    'ner': raw_datasets.remove_columns(['pos_tags', 'pos_tags_numeric', 'dep_tags', 'dep_tags_numeric']),
    'pos': raw_datasets.remove_columns(['ner_tags', 'ner_tags_numeric', 'dep_tags', 'dep_tags_numeric']),
    'dep': raw_datasets.remove_columns(['ner_tags', 'ner_tags_numeric', 'pos_tags', 'pos_tags_numeric']),
}

# create the corresponding task models by supplying the invidual model classes and model configs
multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        'ner': transformers.AutoModelForTokenClassification,
        'pos': transformers.AutoModelForTokenClassification,
        "dep": transformers.AutoModelForTokenClassification,
    },
    model_config_dict={
        'ner': transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(MULTI_LABEL2ID.keys()),
                id2label=MULTI_ID2LABEL,
                label2id=MULTI_LABEL2ID),
        'pos': transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(MULTI_LABEL2ID.keys()),
                id2label=MULTI_ID2LABEL,
                label2id=MULTI_LABEL2ID),
        'dep': transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(MULTI_LABEL2ID.keys()),
                id2label=MULTI_ID2LABEL,
                label2id=MULTI_LABEL2ID),
    },
)

#  convert from raw text to tokenized text inputs
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
convert_func_dict = {
    'ner': convert_to_ner_features,
    'pos': convert_to_pos_features,
    'dep': convert_to_dep_features,
}

columns_dict = {
    'ner': ['input_ids', 'attention_mask', 'labels'],
    'pos': ['input_ids', 'attention_mask', 'labels'],
    'dep': ['input_ids', 'attention_mask', 'labels'],
}

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'tokenizer' : tokenizer}
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
        features_dict[task_name][phase].set_format(
            #type="torch", 
            columns=columns_dict[task_name],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

train_dataset = {
    task_name: dataset['train'] 
    for task_name, dataset in features_dict.items()
}
eval_dataset = {
    task_name: dataset['validation'] 
    for task_name, dataset in features_dict.items()
}

def train(config=None):
  gc.collect()
  torch.cuda.empty_cache()
  with wandb.init(config=config, name=f'{lang}_multitask'):
    # set sweep configuration
    config = wandb.config

    # define training loop    
    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=join('hyperparameter-tuning', 'sweeps', f'{lang}-multitask-sweeps'),
            overwrite_output_dir=True,
            do_train=True,
            report_to='wandb',  # Turn on Weights & Biases logging
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            save_strategy='no',
            evaluation_strategy='epoch'
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset['ner'],
        compute_metrics=compute_metrics
    )
    
    # start training loop
    trainer.train()

#############################################
#                Sweep Setup                #
#############################################
train()