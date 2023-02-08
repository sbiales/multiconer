from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import transformers
import evaluate
import numpy as np
import wandb
import os
from os.path import join

from models import NER_LABEL2ID, NER_ID2LABEL

wandb.login()

data_dir = join(os.getcwd(), 'data')

# Set to None if training multilingual
lang = os.getenv('LANG', '')

model_name = 'xlm-roberta-base'


#############################################
#           Function Definitions            #
#############################################

def convert(obj, classes):
  obj['ner_tags_numeric'] = [classes[t] for t in obj['ner_tags']]
  return obj

def align_labels_with_tokens(labels, word_ids):
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
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, is_split_into_words=True
    )
    all_labels = examples['ner_tags_numeric']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[NER_ID2LABEL[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [NER_ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def model_init():
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=NER_ID2LABEL,
        label2id=NER_LABEL2ID,
    )
    return model

#############################################
#                Model Setup                #
#############################################

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

raw_datasets = raw_datasets.map(convert, fn_kwargs={'classes': NER_LABEL2ID})

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    fn_kwargs={'tokenizer': tokenizer}
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def train(config=None):
  with wandb.init(config=config, entity='sbiales', name=f'{lang}-base'):
    # set sweep configuration
    config = wandb.config

    # Create and set seed to make model reproducible
    seed = int(np.random.rand() * (2**32 - 1))
    print('Seed:', seed)
    transformers.trainer_utils.set_seed(seed)
    wandb.log({'seed': seed})

    # set training arguments
    training_args = TrainingArguments(
        output_dir=join('hyperparameter-tuning', 'sweeps', f'{lang}-base-sweeps'),
	    report_to='wandb',  # Turn on Weights & Biases logging
        do_train=True,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        save_strategy='no',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        remove_unused_columns=False
    )


    # define training loop
    trainer = Trainer(
        # model,
        model_init=model_init,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )


    # start training loop
    trainer.train()

#############################################
#                Sweep Setup                #
#############################################
train()