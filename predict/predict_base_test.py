import transformers
from datasets import load_dataset
import argparse
from os.path import join
import torch
import pydash
from tqdm import tqdm

from models import LANGCODES, NER_LABEL2ID, NER_ID2LABEL

def clean_prediction(labels, word_ids):
    new_labels = []
    current_word = None
    for i, word_id in enumerate(word_ids):
        if word_id != current_word:
            # Start of a new word! Keep the prediction
            if word_id is not None:
              # Check if we skipped a word and fill in with O
              if current_word is not None and word_id - current_word > 1:
                diff = word_id - current_word - 1
                new_labels.extend(['O']*diff)
              # Add the new prediction label
              new_labels.append(labels[i])
            current_word = word_id
    return new_labels

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running the code on', device)

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

    checkpoint = args.checkpoint

    model = transformers.AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        id2label=NER_ID2LABEL,
        label2id=NER_LABEL2ID,
    )
    model.to("cuda:0")
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

    # Prediction
    batched_texts = pydash.arrays.chunk(test_dataset['validation']['tokens'], args.batch_size)
    predictions = []
    for batch_text in tqdm(batched_texts):
        model_inputs = tokenizer(
            batch_text, truncation=True, is_split_into_words=True,
            padding='max_length',
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            out = model(**model_inputs)
        
        # Turn logits into probabilities
        probs = out['logits'].softmax(-1)
        # Get the best fit predicted labels
        pred = probs.argmax(-1)
        # Map IDs to the actual class labels
        pred = [[NER_ID2LABEL[int(p)] for p in prediction] for prediction in pred]

        # Clean off the predictions by only keeping the tokens representing beginnings of words
        for i, prediction in enumerate(pred):
            predictions.append(clean_prediction(prediction, model_inputs.word_ids(i)))
        
    # Write to the predictions file
    with open(join(args.out_dir, f'{lang}.pred.conll'), 'w', encoding='utf-8') as predfile:
        for prediction in predictions:
            predfile.write('\n'.join(prediction))
            predfile.write('\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('-d', '--dataset', type=str, help='The path to the json file to predict for', default=join('data', 'dev.json'))
    # parser.add_argument('-m', '--model', type=str, help='The pretrained model to use', default='xlm-roberta-base')
    parser.add_argument('-c', '--checkpoint', type=str, help='The model checkpoint to use')
    parser.add_argument('-l', '--lang', type=str, help='Which language to predict. If none provided, assume multi')
    parser.add_argument('-o', '--out_dir', type=str, help='The path to put the output files', default=join('predict', 'predictions', 'base'))
    parser.add_argument('-bs', '--batch_size', type=int, help='Prediction batch size.', default=64)

    args = parser.parse_args()

    if args.lang:
        assert args.lang in LANGCODES

    main(args)

