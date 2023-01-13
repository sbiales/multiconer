import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
import argparse
import wandb
from os.path import join

from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
import evaluate

label2id = { 'O': 0, 'B-Station': 1, 'I-Station': 2, 'B-Facility': 3, 'I-Facility': 4, 'B-HumanSettlement': 5, 'I-HumanSettlement': 6, 'B-OtherLOC': 7, 'I-OtherLOC': 8,
 'B-Symptom': 9, 'I-Symptom': 10, 'B-Medication/Vaccine': 11, 'I-Medication/Vaccine': 12, 'B-MedicalProcedure': 13, 'I-MedicalProcedure': 14, 'B-AnatomicalStructure': 15,
 'I-AnatomicalStructure': 16, 'B-Disease': 17, 'I-Disease': 18, 'B-Clothing': 19, 'I-Clothing': 20, 'B-OtherPROD': 21, 'I-OtherPROD': 22, 'B-Vehicle': 23, 'I-Vehicle': 24,
 'B-Food': 25, 'I-Food': 26, 'B-Drink': 27, 'I-Drink': 28, 'B-Artist': 29, 'I-Artist': 30, 'B-Scientist': 31, 'I-Scientist': 32, 'B-OtherPER': 33, 'I-OtherPER': 34,
 'B-Athlete': 35, 'I-Athlete': 36, 'B-SportsManager': 37, 'I-SportsManager': 38, 'B-Politician': 39, 'I-Politician': 40, 'B-Cleric': 41, 'I-Cleric': 42, 'B-ORG': 43, 'I-ORG': 44,
 'B-MusicalGRP': 45, 'I-MusicalGRP': 46, 'B-AerospaceManufacturer': 47, 'I-AerospaceManufacturer': 48, 'B-PublicCorp': 49, 'I-PublicCorp': 50, 'B-SportsGRP': 51, 'I-SportsGRP': 52,
 'B-PrivateCorp': 53, 'I-PrivateCorp': 54, 'B-CarManufacturer': 55, 'I-CarManufacturer': 56, 'B-WrittenWork': 57, 'I-WrittenWork': 58, 'B-MusicalWork': 59, 'I-MusicalWork': 60,
 'B-VisualWork': 61, 'I-VisualWork': 62, 'B-ArtWork': 63, 'I-ArtWork': 64, 'B-Software': 65, 'I-Software': 66,
  # POS tags
  'CCONJ': 67, 'AUX': 68, 'DET': 69, 'VERB': 70, 'SYM': 71, 'INTJ': 72, 'PROPN': 73, 'ADJ': 74, 'NOUN': 75, 'PART': 76, 'X': 77, 'NUM': 78, 'SCONJ': 79, 'ADV': 80, 'PUNCT': 81,
  'ADP': 82, 'PRON': 83,
  # UD tags
  'nmod': 84, 'det': 85, 'discourse:emo': 86, 'vocative:mention': 87, 'cc': 88, 'csubj:pass': 89, 'det:numgov': 90, 'det:poss': 91, 'obj': 92, 'parataxis:discourse': 93,
  'advcl:svc': 94, 'conj:svc': 95, 'expl:subj': 96, 'xcomp:sp': 97, 'obl:npmod': 98, 'mark': 99, 'mark:adv': 100, 'cc:preconj': 101, 'aux': 102, 'advmod:det': 103,
  'parataxis': 104, 'flat:num': 105, 'nummod:gov': 106, 'compound': 107, 'obl': 108, 'expl': 109, 'flat:name': 110, 'obl:agent': 111, 'discourse': 112, 'obl:patient': 113,
  'obj:lvc': 114, 'flat:sibl': 115, 'nmod:tmod': 116, 'goeswith': 117, 'nummod': 118, 'nsubj:pass': 119, 'aux:pass': 120, 'punct': 121, 'aux:tense': 122, 'dislocated': 123,
  'nmod:poss': 124, 'advcl:cleft': 125, 'flat:title': 126, 'parataxis:appos': 127, 'ccomp': 128, 'clf': 129, 'advcl': 130, 'list': 131, 'iobj:agent': 132, 'compound:prt': 133,
  'nsubj': 134, 'flat:range': 135, 'det:nummod': 136, 'amod': 137, 'root': 138, 'aux:caus': 139, 'flat:foreign': 140, 'conj': 141, 'nsubj:caus': 142, 'expl:impers': 143,
  'advmod': 144, 'cop': 145, 'advcl:sp': 146, 'compound:ext': 147, 'det:predet': 148, 'dep': 149, 'expl:pass': 150, 'discourse:sp': 151, 'xcomp': 152, 'obj:agent': 153,
  'obl:mod': 154, 'fixed': 155, 'flat': 156, 'acl:relcl': 157, 'flat:repeat': 158, 'obl:tmod': 159, 'case': 160, 'nmod:npmod': 161, 'vocative': 162, 'appos': 163, 'acl': 164,
  'acl:adv': 165, 'acl:cleft': 166, 'reparandum': 167, 'obl:arg': 168, 'flat:abs': 169, 'iobj': 170, 'csubj': 171, 'orphan': 172, 'dep:comp': 173, 'expl:pv': 174,
  'compound:lvc': 175, 'mark:rel': 176 }
id2label = { i:c for c,i in label2id.items() }

#############################################################################
#                     Multitask model class definition                      #
#############################################################################

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        elif model_class_name.startswith("XLMRoberta"):
            return "roberta"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)


#############################################################################
#                     Conversion/tokenization functions                     #
#############################################################################

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


#############################################################################
#                           Dataloader functions                            #
#############################################################################

class NLPDataCollator:
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """

    def __call__(
        self, features: List[Union[InputDataClass, Dict]]
    ) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict):
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                #if first["labels"].dtype == torch.int64:
                if type(first["labels"][0]) == int:
                    #print(first.items())
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.long
                    )
                else:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.float
                    )
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.tensor([f[k] for f in features])
            return batch
        else:
            # otherwise, revert to using the default collate_batch
            return DataCollator().collate_batch(features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(transformers.Trainer):
    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader

    def get_single_eval_dataloader(self, task_name, eval_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a eval_dataset.")

        eval_sampler = (
            RandomSampler(eval_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(eval_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                sampler=eval_sampler,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return self.get_single_eval_dataloader('ner', self.eval_dataset)


#############################################################################
#                                  Metrics                                  #
#############################################################################

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
    raw_datasets = raw_datasets.map(convert, fn_kwargs={'classes': label2id})

    dataset_dict = {
        'ner': raw_datasets.remove_columns(['pos_tags', 'pos_tags_numeric', 'dep_tags', 'dep_tags_numeric']),
        'pos': raw_datasets.remove_columns(['ner_tags', 'ner_tags_numeric', 'dep_tags', 'dep_tags_numeric']),
        'dep': raw_datasets.remove_columns(['ner_tags', 'ner_tags_numeric', 'pos_tags', 'pos_tags_numeric']),
    }

    # Initialize wandb
    wandb.init(project='thesis', config=args)

    # create the corresponding task models by supplying the invidual model classes and model configs
    model_name = args.model
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict={
            'ner': transformers.AutoModelForTokenClassification,
            'pos': transformers.AutoModelForTokenClassification,
            "dep": transformers.AutoModelForTokenClassification,
        },
        model_config_dict={
            'ner': transformers.AutoConfig.from_pretrained(model_name,
                    num_labels=len(label2id.keys()),
                    id2label=id2label,
                    label2id=label2id),
            'pos': transformers.AutoConfig.from_pretrained(model_name,
                    num_labels=len(label2id.keys()),
                    id2label=id2label,
                    label2id=label2id),
            'dep': transformers.AutoConfig.from_pretrained(model_name,
                    num_labels=len(label2id.keys()),
                    id2label=id2label,
                    label2id=label2id),
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

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=join(args.out_dir, lang + '-multitask'),
            report_to='wandb',
            overwrite_output_dir=True,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            do_train=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            save_steps=3000,
            evaluation_strategy='epoch'
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset['ner'],
        compute_metrics=compute_metrics
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('-t', '--train', type=str, help='The path to the training json file', default='.\\data\\train.json')
    parser.add_argument('-d', '--dev', type=str, help='The path to the dev/validation json file', default='.\\data\\dev.json')
    parser.add_argument('-m', '--model', type=str, help='The model checkpoint to use', default='xlm-roberta-base')
    parser.add_argument('-l', '--lang', type=str, help='Which language to train. If none provided, train on all')
    parser.add_argument('-o', '--out_dir', type=str, help='The path to put the output files', default='.\\results')

    # Training arguments
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size.', default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=2e-5)
    parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs for training.', default=3)
    parser.add_argument('-wd', '--weight_decay', type=float, help='Weight decay', default=0.01)
    #parser.add_argument('-do', '--dropout', type=float, help='Dropout rate', default=0.1)

    args = parser.parse_args()

    #assert args.lang in LANGCODES

    main(args)