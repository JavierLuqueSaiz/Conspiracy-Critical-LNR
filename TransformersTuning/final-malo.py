import pickle
import json
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors

import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel

import pickle

from sklearn.metrics import matthews_corrcoef
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')


def load_dataset_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
  
dataset_test = load_dataset_from_json('dataset_en_official_test_nolabels.json')
dataset_spanish_test = load_dataset_from_json('dataset_es_official_test_nolabels.json')

dataset = load_dataset_from_json('dataset_en_train.json')
dataset_spanish = load_dataset_from_json('dataset_es_train.json')


print("Data loaded\n")

def preprocess(text_list, english=True):
    answer = []
    if english:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set(stopwords.words('spanish'))
    punctuation = set(string.punctuation)
    for text in text_list:
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words and token not in punctuation]
        answer.append(' '.join(tokens))
    return answer

X = [data["text"] for data in dataset]
y = np.array([data["category"] for data in dataset])
X_es = [data["text"] for data in dataset_spanish]
y_es = np.array([data["category"] for data in dataset_spanish])

X_test = [data["text"] for data in dataset_test]
id = np.array([data["id"] for data in dataset_test])
X_es_test = [data["text"] for data in dataset_spanish_test]
id_es = np.array([data["id"] for data in dataset_spanish_test])

token_roberta = RobertaTokenizer.from_pretrained('roberta-base')
model_roberta = RobertaModel.from_pretrained('roberta-base')
token_roberta_es = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model_roberta_es = RobertaModel.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')

def get_sentence_rep2(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vector = outputs[0][:, 0, :]  # Get the CLS vector
    return cls_vector

print("Spanish vectorization")
roberta_es_test = [get_sentence_rep2(text, model_roberta_es, token_roberta_es) for text in X_es_test]
roberta_es = [get_sentence_rep2(text, model_roberta_es, token_roberta_es) for text in X_es]
print("COMPLETED")

print("RoBERTa loaded")

vectorizer = CountVectorizer(binary=True)
bbw_en = vectorizer.transform(preprocess(X))
bbw_en_test = vectorizer.transform(preprocess(X_test))

print("Binary Bag of Words loaded")

r = np.array(roberta_es).shape[0]
c = np.array(roberta_es).shape[2]
new_roberta_es = np.array(roberta_es).reshape(r,c)

r = np.array(roberta_es_test).shape[0]
c = np.array(roberta_es_test).shape[2]
new_roberta_es_test = np.array(roberta_es_test).reshape(r,c)

model1_es = svm.SVC(kernel="linear")
model2_es = LogisticRegression(penalty=None)
model3_es = DecisionTreeClassifier(criterion="log_loss")
model4_es = KNeighborsClassifier()
base_models_es = [('dt', model3_es),
                ('svm', model1_es),
                ('knn', model4_es)]
ensemble_es = StackingClassifier(
    estimators = base_models_es,
    final_estimator = model2_es
)
ensemble_es.fit(new_roberta_es, y_es)
predicted_en = ensemble_es.predict(new_roberta_es_test)

model1_es = svm.SVC(kernel="linear")
model2_es = LogisticRegression(penalty="l2")
model3_es = DecisionTreeClassifier(criterion="gini")
ensemble_es = VotingClassifier(
    estimators=[('lr', model2_es),
                ('dt', model3_es),
                ('svm', model1_es)],
                voting='hard')
ensemble_es.fit(bbw_en, y)
predicted_es = ensemble_es.predict(bbw_en_test)

print(predicted_es)

predictions_df_es = pd.DataFrame()
predictions_df_en = pd.DataFrame()

predictions_df_es["id"] = id_es
predictions_df_en["id"] = id

predictions_df_es["majority vote"] = predicted_es
predictions_df_en["majority vote"] = predicted_en

json_output = []
for index, row in predictions_df_es.iterrows():
    json_output.append({"id": row['id'], "category": row['majority_vote']})

json_output_path = "lnr-luqrud_task1_es.json"
with open(json_output_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=2)

# Display DataFrame to user (for Jupyter Notebook environment)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Predictions DataFrame", dataframe=predictions_df)

print(f"JSON output has been saved to {json_output_path}")

json_output = []
for index, row in predictions_df_en.iterrows():
    json_output.append({"id": row['id'], "category": row['majority_vote']})

json_output_path = "lnr-luqrud_task1_en.json"
with open(json_output_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=2)

# Display DataFrame to user (for Jupyter Notebook environment)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Predictions DataFrame", dataframe=predictions_df)

print(f"JSON output has been saved to {json_output_path}")
