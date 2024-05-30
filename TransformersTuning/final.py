import pandas as pd
from datareader import en_train_df, es_train_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split, KFold
from fine_tuning import training, validate
SEED=1234
from utils import set_seed, product_dict
import wandb
from datetime import datetime

TRAIN_DATASET_ES="dataset_es_official_test_nolabels.json"
TRAIN_DATASET_EN="dataset_en_official_test_nolabels.json"


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

import wandb
run = wandb.init()

# Define SEED for reproducibility
SEED = 1234

num_samples = len(es_test_df)
fold_predictions = np.zeros((num_samples, num_folds))

all_predictions = []

artifacts = ['javier-luque/LNR_2024-05-30_14-03-15_FIN/run-n27mn2wt-history:v0',
            'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-21e45ao9-history:v0',
            'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-7f41zx51-history:v0',
            'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-pluwlcjo-history:v0',
            'javier-luque/LNR_2024-05-30_14-03-15_FIN/run-b0ivyb1p-history:v0']

for fold in artifacts:
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

artifacts = []

for i in artifacts:
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
