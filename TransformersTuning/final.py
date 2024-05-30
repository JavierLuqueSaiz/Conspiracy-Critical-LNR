import pandas as pd
from datareader import en_train_df, es_train_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split, KFold
from fine_tuning import training
SEED=1234
from utils import set_seed, product_dict
import wandb
from datetime import datetime
import pandas as pd
import json
import os
import torch
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam, RMSprop
from transformers  import  get_scheduler
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import KFold
import wandb
from datetime import datetime
import os
import shutil
import random
import numpy as np
from itertools import product

# utils.py
def set_seed(seed):
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments
    Parameters:
    seed: the number to set the seed to
    Return: None
    """
    # Random seed
    random.seed(seed)
    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def remove_previous_model(folder):
    dirs = [x for x in os.listdir(folder) if os.path.isdir(folder+os.sep+x)]
    for x in dirs:
        shutil.rmtree(folder+os.sep+x, ignore_errors=False, onerror=None)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

# mydataset.py
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, mode='train'):
        self.encodings = encodings
        if labels is None:
            self.labels = None  # No labels present
        else:
            self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx]).clone().detach()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])  # Or another key that's always present
    
# datareader.py
import pandas as pd
import json

BINARY_MAPPING_CRITICAL_POS = {'CONSPIRACY': 0, 'CRITICAL': 1}
BINARY_MAPPING_CONSPIRACY_POS = {'CRITICAL': 0, 'CONSPIRACY': 1}

CATEGORY_MAPPING_CRITICAL_POS_INVERSE = {0: 'CONSPIRACY', 1: 'CRITICAL'}
CATEGORY_MAPPING_CONSPIRACY_POS_INVERSE = {0: 'CRITICAL', 1: 'CONSPIRACY'}

# TRAIN_DATASET_ES="/kaggle/input/dataset-oppositional/dataset_oppositional/training/dataset_oppositional/dataset_es_train.json"
# TRAIN_DATASET_EN="/kaggle/input/dataset-oppositional/dataset_oppositional/training/dataset_oppositional/dataset_en_train.json"
TEST_DATASET_EN ="dataset_en_official_test_nolabels.json"
TEST_DATASET_ES ="dataset_es_official_test_nolabels.json"


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
#             classes = [binmap[doc['category']] for doc in dataset]
#             classes = pd.Series(classes)
        ids = pd.Series([doc['id'] for doc in dataset])
#         data = pd.DataFrame({"text": texts, "id": ids, "label": classes})
        data = pd.DataFrame({"text": texts, "id": ids})

        return data


myReader=PAN24Reader()
es_test_df = myReader.load_dataset_classification(TEST_DATASET_ES, string_labels=False, positive_class='conspiracy')
en_test_df = myReader.load_dataset_classification(TEST_DATASET_EN, string_labels=False, positive_class='conspiracy')

BINARY_MAPPING_CRITICAL_POS = {'CONSPIRACY': 0, 'CRITICAL': 1}
BINARY_MAPPING_CONSPIRACY_POS = {'CRITICAL': 0, 'CONSPIRACY': 1}

CATEGORY_MAPPING_CRITICAL_POS_INVERSE = {0: 'CONSPIRACY', 1: 'CRITICAL'}
CATEGORY_MAPPING_CONSPIRACY_POS_INVERSE = {0: 'CRITICAL', 1: 'CONSPIRACY'}

TRAIN_DATASET_ES="dataset_es_official_test_nolabels.json"
TRAIN_DATASET_EN="dataset_en_official_test_nolabels.json"

# Define the validation function
def validate(_wandb, _model, _test_data, _tokenizer, _batch_size=32, _padding="max_length", _max_length=512, _truncation=True, _measure="accuracy", evaltype=False):
    test_encodings = _tokenizer(_test_data['text'].tolist(), max_length=_max_length, truncation=_truncation, padding=_padding, return_tensors="pt")
    
    # Comprobar si hay etiquetas en los datos, asumir que no si no están presentes
    if 'label' in _test_data:
        labels = _test_data['label'].tolist()
    else:
        labels = None  # No labels provided
    
    test = MyDataset(test_encodings, labels, mode="predict" if labels is None else "train")
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=_batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    _model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Usando {torch.cuda.device_count()} GPUs")
        _model = torch.nn.DataParallel(_model)

    _model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = _model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.append(predictions.cpu())  # Move predictions to CPU

    # Concatenar todas las predicciones
    all_predictions = torch.cat(all_predictions).numpy()  # Ensure tensor is on CPU before converting to numpy
    return all_predictions

import wandb
run = wandb.init()

# Define SEED for reproducibility
SEED = 1234
num_folds=5
num_samples = len(es_test_df)
fold_predictions = np.zeros((num_samples, num_folds))

all_predictions = []

artifacts = ['javier-luque/LNR_2024-05-30_14-03-15_FIN/run-n27mn2wt-history:v0',
            'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-21e45ao9-history:v0',
            'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-7f41zx51-history:v0',
            'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-pluwlcjo-history:v0',
            'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-b0ivyb1p-history:v0']

for fold, i in enumerate(artifacts):
    artifact = run.use_artifact(i, type='wandb-history')
    artifact_dir = artifact.download()
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model.eval()

    predictions = validate(_wandb=wandb, _model=model, _test_data=es_test_df, _tokenizer=tokenizer, _batch_size=64, _padding="max_length", _max_length=128, _truncation=True, _measure="mcc", evaltype=False)

    fold_predictions[:, fold] = predictions

# Compute the majority vote for each sample
final_preds = [Counter(fold_predictions[i, :]).most_common(1)[0][0] for i in range(num_samples)]

# Convert numerical predictions to labels
def transform_label(pred):
    return "CONSPIRACY" if pred == 1 else "CRITICAL"

# Create DataFrame with predictions
predictions_df = pd.DataFrame(fold_predictions, columns=[f'fold_{i}' for i in range(num_folds)])
predictions_df = predictions_df.map(transform_label)
predictions_df['majority_vote'] = list(map(transform_label, final_preds))
predictions_df['id'] = es_test_df['id']

# Save DataFrame to CSV
output_csv_path = "es_test_predictions.csv"
predictions_df.to_csv(output_csv_path, index=False)

# Create JSON with the required format
json_output = []
for index, row in predictions_df.iterrows():
    json_output.append({"id": row['id'], "category": row['majority_vote']})

json_output_path = "es_test_predictions.json"
with open(json_output_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=2)

# Display DataFrame to user (for Jupyter Notebook environment)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Predictions DataFrame", dataframe=predictions_df)

print(f"Predictions have been saved to {output_csv_path}")
print(f"JSON output has been saved to {json_output_path}")

# Define SEED for reproducibility
SEED = 1234

num_samples = len(es_test_df)
fold_predictions = np.zeros((num_samples, num_folds))

all_predictions = []

artifacts = ['javier-luque/LNR_2024-05-30_14-03-15_FIN/run-xhprw64q-history:v0',
             'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-50dbooo4-history:v0',
             'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-xxveo7fe-history:v0',
             'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-egd7d5bb-history:v0',
             'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-1tamhoak-history:v0']

for fold, i in enumerate(artifacts):
    artifact = run.use_artifact(i, type='wandb-history')
    artifact_dir = artifact.download()
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model.eval()

    predictions = validate(_wandb=wandb, _model=model, _test_data=en_test_df, _tokenizer=tokenizer, _batch_size=64, _padding="max_length", _max_length=128, _truncation=True, _measure="mcc", evaltype=False)

    fold_predictions[:, fold] = predictions

# Compute the majority vote for each sample
final_preds = [Counter(fold_predictions[i, :]).most_common(1)[0][0] for i in range(num_samples)]

# Convert numerical predictions to labels
def transform_label(pred):
    return "CONSPIRACY" if pred == 1 else "CRITICAL"

# Create DataFrame with predictions
predictions_df = pd.DataFrame(fold_predictions, columns=[f'fold_{i}' for i in range(num_folds)])
predictions_df = predictions_df.map(transform_label)
predictions_df['majority_vote'] = list(map(transform_label, final_preds))
predictions_df['id'] = en_test_df['id']

# Save DataFrame to CSV
output_csv_path = "en_test_predictions.csv"
predictions_df.to_csv(output_csv_path, index=False)

# Create JSON with the required format
json_output = []
for index, row in predictions_df.iterrows():
    json_output.append({"id": row['id'], "category": row['majority_vote']})

json_output_path = "en_test_predictions.json"
with open(json_output_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=2)

# Display DataFrame to user (for Jupyter Notebook environment)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Predictions DataFrame", dataframe=predictions_df)

print(f"Predictions have been saved to {output_csv_path}")
print(f"JSON output has been saved to {json_output_path}")
