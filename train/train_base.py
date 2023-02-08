import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import transformers
import evaluate
import numpy as np
import wandb

from models import LANGCODES, NER_LABEL2ID, NER_ID2LABEL

def main(args):
    # Load the train and dev dataset files into a HuggingFace dataset
    raw_datasets = load_dataset('json', data_files={
    'train': args.train, 
    'validation': args.dev,
    },
    chunksize=40<<20,
    ignore_verifications=True)
    if args.lang:
        raw_datasets = raw_datasets.filter(lambda example: example["domain"] == args.lang)
    raw_datasets = raw_datasets.map(convert, fn_kwargs={'classes': NER_LABEL2ID})

    # Tokenize the datasets and align labels
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={'tokenizer': tokenizer}
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        id2label=NER_ID2LABEL,
        label2id=NER_LABEL2ID,
    )

    # Initialize wandb
    wandb.init(project="thesis", config=args)

    if(args.seed):
        seed = args.seed
    else:
        seed = int(np.random.rand() * (2**32 - 1))
    print('Seed:', seed)
    transformers.trainer_utils.set_seed(seed)
    wandb.log({'seed': seed})

    training_args = TrainingArguments(
        "base-finetuned-ner",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        push_to_hub=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to='wandb'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    trainer.train()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('-t', '--train', type=str, help='The path to the training json file', default='.\\data\\train.json')
    parser.add_argument('-d', '--dev', type=str, help='The path to the dev/validation json file', default='.\\data\\dev.json')
    parser.add_argument('-m', '--model', type=str, help='The model checkpoint to use', default='xlm-roberta-base')
    parser.add_argument('-l', '--lang', type=str, help='Which language to train. If none provided, train on all')
    parser.add_argument('-s', '--seed', type=int, help='Seed for the model')

    # Training arguments
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size.', default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=2e-5)
    parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs for training.', default=3)
    parser.add_argument('-wd', '--weight_decay', type=float, help='Weight decay', default=0.01)
    parser.add_argument('-wr', '--warmup_ratio', type=float, help='Warmup ratio', default=0.0)
    #parser.add_argument('-do', '--dropout', type=float, help='Dropout rate', default=0.1)

    args = parser.parse_args()

    if args.lang:
        assert args.lang in LANGCODES

    main(args)