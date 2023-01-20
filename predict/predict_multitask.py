import numpy as np
import transformers
from datasets import load_dataset
import argparse
import wandb
from os.path import join
import torch

import evaluate

from models import MULTI_LABEL2ID, MULTI_ID2LABEL, LANGCODES
from models import MultitaskModel, NLPDataCollator, MultitaskTrainer, DataLoaderWithTaskname, convert_to_ner_features, convert_to_pos_features, convert_to_dep_features

def convert(obj, classes):
  obj['ner_tags_numeric'] = [classes[t] for t in obj['ner_tags']]
  obj['pos_tags_numeric'] = [classes[t] for t in obj['pos_tags']]
  obj['dep_tags_numeric'] = [classes[t] for t in obj['dep_tags']]
  return obj

def main(args):
    # Load the dataset file into a HuggingFace dataset
    raw_datasets = load_dataset('json', data_files={
    'validation': args.dataset
    },
    chunksize=40<<20,
    ignore_verifications=True)
    if args.lang:
        raw_datasets = raw_datasets.filter(lambda example: example['domain'] == args.lang)
        lang = args.lang
    else:
        lang = 'multi'
    raw_datasets = raw_datasets.map(convert, fn_kwargs={'classes': MULTI_LABEL2ID})

    model_name = args.model

    # Create the dictionaries based on which tasks are being predicted
    dataset_dict = {
        'ner': raw_datasets.remove_columns(['pos_tags', 'pos_tags_numeric', 'dep_tags', 'dep_tags_numeric']),
    }
    model_type_dict={
        'ner': transformers.AutoModelForTokenClassification
    }
    model_config_dict={
        'ner': transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(MULTI_LABEL2ID.keys()),
                id2label=MULTI_ID2LABEL,
                label2id=MULTI_LABEL2ID)
    }
    convert_func_dict = {
        'ner': convert_to_ner_features
    }
    columns_dict = {
        'ner': ['input_ids', 'attention_mask', 'labels']
    }

    if 'P' in args.tasks:
        dataset_dict['pos'] = raw_datasets.remove_columns(['ner_tags', 'ner_tags_numeric', 'dep_tags', 'dep_tags_numeric'])
        model_type_dict['pos'] = transformers.AutoModelForTokenClassification
        model_config_dict['pos'] = transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(MULTI_LABEL2ID.keys()),
                id2label=MULTI_ID2LABEL,
                label2id=MULTI_LABEL2ID)
        convert_func_dict['pos'] = convert_to_pos_features
        columns_dict['pos'] = ['input_ids', 'attention_mask', 'labels']

    if 'D' in args.tasks:
        dataset_dict['dep'] = raw_datasets.remove_columns(['ner_tags', 'ner_tags_numeric', 'pos_tags', 'pos_tags_numeric'])
        model_type_dict['dep'] = transformers.AutoModelForTokenClassification
        model_config_dict['dep'] = transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(MULTI_LABEL2ID.keys()),
                id2label=MULTI_ID2LABEL,
                label2id=MULTI_LABEL2ID)
        convert_func_dict['dep'] = convert_to_dep_features
        columns_dict['dep'] = ['input_ids', 'attention_mask', 'labels']

    # Initialize wandb
    #wandb.init(project='thesis', config=args)

    # create the corresponding task models by supplying the invidual model classes and model configs
    checkpoint = args.checkpoint
    #config = MultitaskConfig.from_json_file(join(checkpoint, 'config.json'))
    #multitask_model = MultitaskModel.load(config)
    
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict=model_type_dict,
        model_config_dict=model_config_dict
    )

    state_dict = torch.load(join(checkpoint, 'pytorch_model.bin'))
    multitask_model.load_state_dict(state_dict)

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
            features_dict[task_name][phase].set_format(
                #type="torch", 
                columns=columns_dict[task_name],
            )

    eval_dataset = {
        task_name: dataset['validation'] 
        for task_name, dataset in features_dict.items()
    }

    trainer = MultitaskTrainer(
        args=transformers.TrainingArguments(
            output_dir=args.out_dir),
        model=multitask_model,
        data_collator=NLPDataCollator(),
        eval_dataset=eval_dataset['ner']
    )

    eval_dataloader = DataLoaderWithTaskname(
        'ner',
        trainer.get_eval_dataloader(eval_dataset=features_dict['ner']["validation"])
    )

    # Prediction
    preds = trainer.evaluation_loop(
        eval_dataloader, 
        description=f"Predictions: {'ner'}",
    )
    logits, labels, metrics, numsamples = preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_predictions = [
        [MULTI_ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)]
    
    # Comment this out if you don't want to print all of the prediction arrays to the console
    for prediction in true_predictions:
        print(prediction)
    
    # Evaluation (if dev set not test)
    metric = evaluate.load("seqeval")
    true_labels = [[MULTI_ID2LABEL[l] for l in label if l != -100] for label in labels]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    print({
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    })

    # Write to the predictions file
    with open(join(args.out_dir, f'{lang}.pred.conll'), 'w', encoding='utf-8') as predfile:
        for prediction in true_predictions:
            predfile.write('\n'.join(prediction))
            predfile.write('\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('-d', '--dataset', type=str, help='The path to the json file to predict for', default=join('data', 'dev.json'))
    parser.add_argument('-m', '--model', type=str, help='The pretrained model to use', default='xlm-roberta-base')
    parser.add_argument('-c', '--checkpoint', type=str, help='The model checkpoint to use')
    parser.add_argument('-l', '--lang', type=str, help='Which language to predict. If none provided, assume multi')
    parser.add_argument('-o', '--out_dir', type=str, help='The path to put the output files', default=join('predict', 'predictions'))
    parser.add_argument('-t', '--tasks', type=str, nargs = '*', choices=['D', 'P'], help='Which tasks to include (P for POS, D for dependency relations)')

    args = parser.parse_args()

    if args.lang:
        assert args.lang in LANGCODES

    main(args)

