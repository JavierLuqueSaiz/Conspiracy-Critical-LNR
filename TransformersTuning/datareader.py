#!/usr/bin/python3
import pandas as pd
import json

BINARY_MAPPING_CRITICAL_POS = {'CONSPIRACY': 0, 'CRITICAL': 1}
BINARY_MAPPING_CONSPIRACY_POS = {'CRITICAL': 0, 'CONSPIRACY': 1}

CATEGORY_MAPPING_CRITICAL_POS_INVERSE = {0: 'CONSPIRACY', 1: 'CRITICAL'}
CATEGORY_MAPPING_CONSPIRACY_POS_INVERSE = {0: 'CRITICAL', 1: 'CONSPIRACY'}

TRAIN_DATASET_ES="./dataset_oppositional/training/dataset_oppositional/dataset_es_train.json"
TRAIN_DATASET_EN="./dataset_oppositional/training/dataset_oppositional/dataset_en_train.json"
#TEST_DATASET_EN ="./dataset_oppositional/test/dataset_oppositional_test_nolabels/dataset_en_official_test_nolabels.json"
#TEST_DATASET_ES ="./dataset_oppositional/test/dataset_oppositional_test_nolabels/dataset_en_official_test_nolabels.json"


class PAN24Reader:
    def __init__(self):
        pass
    def read_json_file(self, path):
        dataset=[]
        print(f'Loading official JSON {path} dataset')
        with open(path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        return dataset

    def load_dataset_classification(self, path, string_labels=False, positive_class='conspiracy'):
        dataset = self.read_json_file(path)
        # convert to a format suitable for classification
        texts = pd.Series([doc['text'] for doc in dataset])
        if string_labels:
            classes = pd.Series([doc['category'] for doc in dataset])
        else:
            if positive_class == 'conspiracy':
                binmap = BINARY_MAPPING_CONSPIRACY_POS
            elif positive_class == 'critical':
                binmap = BINARY_MAPPING_CRITICAL_POS
            else:
                raise ValueError(f'Unknown positive class: {positive_class}')
            classes = [binmap[doc['category']] for doc in dataset]
            classes = pd.Series(classes)
        ids = pd.Series([doc['id'] for doc in dataset])
        data = pd.DataFrame({"text": texts, "id": ids, "label": classes})
        return data


myReader=PAN24Reader()
es_train_df = myReader.load_dataset_classification(TRAIN_DATASET_ES, string_labels=False, positive_class='conspiracy')
en_train_df = myReader.load_dataset_classification(TRAIN_DATASET_EN, string_labels=False, positive_class='conspiracy')
