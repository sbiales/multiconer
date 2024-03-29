import numpy as np
import transformers
from datasets import load_dataset
import argparse
import wandb
from os.path import join

import evaluate

from models import NER_LABEL2ID, NER_ID2LABEL, POS_LABEL2ID, POS_ID2LABEL, DEP_LABEL2ID, DEP_ID2LABEL, LANGCODES
from models import MultitaskModel, NLPDataCollator, MultitaskTrainer, convert_to_ner_features, convert_to_pos_features, convert_to_dep_features


#############################################################################
#                            Conversion function                            #
#############################################################################

def convert(obj, classes, tasks):
  obj['ner_tags_numeric'] = [classes['ner'][t] for t in obj['ner_tags']]
  if 'P' in tasks:
    obj['pos_tags_numeric'] = [classes['pos'][t] for t in obj['pos_tags']]
  if 'D' in tasks:
    obj['dep_tags_numeric'] = [classes['dep'][t] for t in obj['dep_tags']]
  return obj


#############################################################################
#                                  Metrics                                  #
#############################################################################

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


#############################################################################
#                                   Main                                    #
#############################################################################

def main(args):
    # Load the train and dev dataset files into a HuggingFace dataset
    raw_datasets = load_dataset('json', data_files={
    'train': args.train, 
    'validation': args.dev,
    },
    chunksize=40<<20,
    ignore_verifications=True)
    if args.lang:
        raw_datasets = raw_datasets.filter(lambda example: example['domain'] == args.lang)
        lang = args.lang
    else:
        lang = 'multi'
        if 'D' in args.tasks:
            raw_datasets = raw_datasets.filter(lambda example: example["domain"] != 'bn')
    classes = { 'ner': NER_LABEL2ID, 'pos': POS_LABEL2ID, 'dep': DEP_LABEL2ID }
    raw_datasets = raw_datasets.map(convert, fn_kwargs={'classes': classes, 'tasks': args.tasks})

    # Initialize wandb
    wandb.init(project='thesis', entity='sbiales', config=args)

    # Set seed and make model deterministic
    if(args.seed):
        seed = args.seed
    else:
        seed = int(np.random.rand() * (2**32 - 1))
    print('Seed:', seed)
    print('Tasks:', args.tasks)
    transformers.trainer_utils.enable_full_determinism(seed)
    wandb.log({
        'seed': seed,
        'tasks': args.tasks
    })

    # Create the dictionaries based on which tasks are being predicted
    model_name = args.model
    dataset_dict = {
        'ner': raw_datasets,
    }
    model_type_dict={
        'ner': transformers.AutoModelForTokenClassification
    }
    model_config_dict={
        'ner': transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(NER_LABEL2ID.keys()),
                id2label=NER_ID2LABEL,
                label2id=NER_LABEL2ID)
    }
    convert_func_dict = {
        'ner': convert_to_ner_features
    }
    columns_dict = {
        'ner': ['input_ids', 'attention_mask', 'labels']
    }

    if 'P' in args.tasks:
        dataset_dict['pos'] = raw_datasets
        model_type_dict['pos'] = transformers.AutoModelForTokenClassification
        model_config_dict['pos'] = transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(POS_LABEL2ID.keys()),
                id2label=POS_ID2LABEL,
                label2id=POS_LABEL2ID)
        convert_func_dict['pos'] = convert_to_pos_features
        columns_dict['pos'] = ['input_ids', 'attention_mask', 'labels']

    if 'D' in args.tasks:
        dataset_dict['dep'] = raw_datasets
        model_type_dict['dep'] = transformers.AutoModelForTokenClassification
        model_config_dict['dep'] = transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(DEP_LABEL2ID.keys()),
                id2label=DEP_ID2LABEL,
                label2id=DEP_LABEL2ID)
        convert_func_dict['dep'] = convert_to_dep_features
        columns_dict['dep'] = ['input_ids', 'attention_mask', 'labels']

    # create the corresponding task models by supplying the invidual model classes and model configs
    def model_init():
        model = MultitaskModel.create(
            model_name=model_name,
            model_type_dict=model_type_dict,
            model_config_dict=model_config_dict,
        )
        return model

    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict=model_type_dict,
        model_config_dict=model_config_dict,
    )

    #  convert from raw text to tokenized text inputs
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

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
    eval_dataset = features_dict['ner']['validation']

    trainer = MultitaskTrainer(
        #model=multitask_model,
        model_init=model_init,
        args=transformers.TrainingArguments(
            output_dir=join(args.out_dir, f'{lang}-multitask-{"".join(args.tasks)}'),
            report_to='wandb',
            overwrite_output_dir=True,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            do_train=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=16,
            save_strategy='epoch',
            save_total_limit=3,
            evaluation_strategy='steps',
            eval_steps=2000,
            full_determinism=True,
            label_names=['labels']
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('-t', '--train', type=str, help='The path to the training json file', default=join('data', 'train.json'))
    parser.add_argument('-d', '--dev', type=str, help='The path to the dev/validation json file', default=join('data', 'dev.json'))
    parser.add_argument('-m', '--model', type=str, help='The model checkpoint to use', default='xlm-roberta-base')
    parser.add_argument('-l', '--lang', type=str, help='Which language to train. If none provided, train on all')
    parser.add_argument('-o', '--out_dir', type=str, help='The path to put the output files', default='checkpoints')
    parser.add_argument('--tasks', type=str, nargs = '*', choices=['D', 'P'], help='Which tasks to include (P for POS, D for dependency relations)')
    parser.add_argument('-s', '--seed', type=int, help='Seed for the model')

    # Training arguments
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size.', default=8)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=2e-5)
    parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs for training.', default=3)
    parser.add_argument('-wd', '--weight_decay', type=float, help='Weight decay', default=0.01)
    parser.add_argument('-wr', '--warmup_ratio', type=float, help='Warmup ratio', default=0.0)
    #parser.add_argument('-do', '--dropout', type=float, help='Dropout rate', default=0.1)

    args = parser.parse_args()

    if args.lang:
        assert args.lang in LANGCODES

    main(args)