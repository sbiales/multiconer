import transformers
from datasets import load_dataset
import argparse
from os.path import join
import torch
import pydash
from tqdm import tqdm


from models import MULTI_LABEL2ID, MULTI_ID2LABEL, LANGCODES
from models import MultitaskModel

def main(args):
    # Load the dataset file into a HuggingFace dataset
    test_dataset = load_dataset('json', data_files={
    'validation': args.dataset
    },
    chunksize=40<<20,
    ignore_verifications=True)
    if args.lang:
        lang = args.lang
    else:
        lang = 'multi'

    model_name = args.model

    # Create the dictionaries based on which tasks are being predicted

    model_type_dict={
        'ner': transformers.AutoModelForTokenClassification
    }
    model_config_dict={
        'ner': transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(MULTI_LABEL2ID.keys()),
                id2label=MULTI_ID2LABEL,
                label2id=MULTI_LABEL2ID)
    }

    if 'P' in args.tasks:
        model_type_dict['pos'] = transformers.AutoModelForTokenClassification
        model_config_dict['pos'] = transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(MULTI_LABEL2ID.keys()),
                id2label=MULTI_ID2LABEL,
                label2id=MULTI_LABEL2ID)

    if 'D' in args.tasks:
        model_type_dict['dep'] = transformers.AutoModelForTokenClassification
        model_config_dict['dep'] = transformers.AutoConfig.from_pretrained(model_name,
                num_labels=len(MULTI_LABEL2ID.keys()),
                id2label=MULTI_ID2LABEL,
                label2id=MULTI_LABEL2ID)

    # Initialize wandb
    #wandb.init(project='thesis', config=args)

    # Load the model
    checkpoint = args.checkpoint
    
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict=model_type_dict,
        model_config_dict=model_config_dict
    )

    state_dict = torch.load(join(checkpoint, 'pytorch_model.bin'))
    multitask_model.load_state_dict(state_dict)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # Prediction
    batched_texts = pydash.arrays.chunk(test_dataset['validation']['tokens'], args.batch_size)
    predictions = []
    for batch_text in tqdm(batched_texts):
        model_inputs = tokenizer(
            batch_text, truncation=True, is_split_into_words=True,
            padding='max_length',
            return_tensors='pt'
        )
        out = multitask_model('ner', **model_inputs)
        
        # Turn logits into probabilities
        probs = out['logits'].softmax(-1)
        # Get the best fit predicted labels
        pred = probs.argmax(-1)
        # Map IDs to the actual class labels
        pred = [[MULTI_ID2LABEL[int(p)] for p in prediction] for prediction in pred]

        # Clean off the predictions by only keeping the tokens representing beginnings of words
        if model_name.startswith('xlm-roberta'):
            for i, prediction in enumerate(pred):
                predictions.append([p for p, t in zip(prediction, model_inputs.tokens(i)) if t.startswith('▁')])
        else:
            print('Please implement code to handle cleaning the predictions based on your model tokenizer')
            predictions.extend(pred)
    #print(predictions)

    # Write to the predictions file
    with open(join(args.out_dir, f'{lang}.pred.conll'), 'w', encoding='utf-8') as predfile:
        for prediction in predictions:
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
    parser.add_argument('-t', '--tasks', type=str, nargs = '*', choices=['D', 'P'], help='Which tasks the model was trained on (P for POS, D for dependency relations)')
    parser.add_argument('-bs', '--batch_size', type=int, help='Prediction batch size.', default=16)

    args = parser.parse_args()

    if args.lang:
        assert args.lang in LANGCODES

    main(args)
