import pandas as pd
from datareader import en_train_df, es_train_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split, KFold
from fine_tuning import training, validate
SEED=1234
from utils import set_seed, product_dict
import wandb
from datetime import datetime

es_test_df = myReader.load_dataset_classification(TEST_DATASET_ES, string_labels=False, positive_class='conspiracy')
en_test_df = myReader.load_dataset_classification(TEST_DATASET_EN, string_labels=False, positive_class='conspiracy')

import wandb
run = wandb.init()
artifacts = ['javier-luque/LNR_2024-05-30_14-03-15_FIN/run-n27mn2wt-history:v0']
for i in artifacts:
    artifact = run.use_artifact(i, type='wandb-history')
    artifact_dir = artifact.download()
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
