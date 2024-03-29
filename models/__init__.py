from models.constants import NER_LABEL2ID, NER_ID2LABEL, POS_LABEL2ID, POS_ID2LABEL, DEP_LABEL2ID, DEP_ID2LABEL, MULTI_LABEL2ID, MULTI_ID2LABEL, LANGCODES
from models.multitask import MultitaskModel, NLPDataCollator, StrIgnoreDevice, DataLoaderWithTaskname, MultitaskDataloader, MultitaskTrainer
from models.multitask import convert_to_ner_features, convert_to_pos_features, convert_to_dep_features