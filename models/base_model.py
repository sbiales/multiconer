import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import wandb

label2id = { 'O': 0, 'B-Station': 1, 'I-Station': 2, 'B-Facility': 3, 'I-Facility': 4, 'B-HumanSettlement': 5, 'I-HumanSettlement': 6, 'B-OtherLOC': 7, 'I-OtherLOC': 8,
 'B-Symptom': 9, 'I-Symptom': 10, 'B-Medication/Vaccine': 11, 'I-Medication/Vaccine': 12, 'B-MedicalProcedure': 13, 'I-MedicalProcedure': 14, 'B-AnatomicalStructure': 15,
 'I-AnatomicalStructure': 16, 'B-Disease': 17, 'I-Disease': 18, 'B-Clothing': 19, 'I-Clothing': 20, 'B-OtherPROD': 21, 'I-OtherPROD': 22, 'B-Vehicle': 23, 'I-Vehicle': 24,
 'B-Food': 25, 'I-Food': 26, 'B-Drink': 27, 'I-Drink': 28, 'B-Artist': 29, 'I-Artist': 30, 'B-Scientist': 31, 'I-Scientist': 32, 'B-OtherPER': 33, 'I-OtherPER': 34,
 'B-Athlete': 35, 'I-Athlete': 36, 'B-SportsManager': 37, 'I-SportsManager': 38, 'B-Politician': 39, 'I-Politician': 40, 'B-Cleric': 41, 'I-Cleric': 42, 'B-ORG': 43, 'I-ORG': 44,
 'B-MusicalGRP': 45, 'I-MusicalGRP': 46, 'B-AerospaceManufacturer': 47, 'I-AerospaceManufacturer': 48, 'B-PublicCorp': 49, 'I-PublicCorp': 50, 'B-SportsGRP': 51, 'I-SportsGRP': 52,
 'B-PrivateCorp': 53, 'I-PrivateCorp': 54, 'B-CarManufacturer': 55, 'I-CarManufacturer': 56, 'B-WrittenWork': 57, 'I-WrittenWork': 58, 'B-MusicalWork': 59, 'I-MusicalWork': 60,
 'B-VisualWork': 61, 'I-VisualWork': 62, 'B-ArtWork': 63, 'I-ArtWork': 64, 'B-Software': 65, 'I-Software': 66 }
id2label = { i:c for c,i in label2id.items() }
LANGCODES = ['bn', 'de', 'en', 'es', 'fa', 'fr', 'hi', 'it', 'pt', 'sv', 'uk', 'zh']

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
    raw_datasets = raw_datasets.map(convert, fn_kwargs={'classes': label2id})

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
        id2label=id2label,
        label2id=label2id,
    )

    # Initialize wandb
    wandb.init(project="thesis", config=args)

    training_args = TrainingArguments(
        "base-finetuned-ner",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
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
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
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

    # Training arguments
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size.', default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=2e-5)
    parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs for training.', default=3)
    parser.add_argument('-wd', '--weight_decay', type=float, help='Weight decay', default=0.01)
    #parser.add_argument('-do', '--dropout', type=float, help='Dropout rate', default=0.1)

    args = parser.parse_args()

    assert args.lang in LANGCODES

    main(args)