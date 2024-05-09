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

nltk.download('stopwords')
nltk.download('punkt')

def load_dataset_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
  
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
y = [data["category"] for data in dataset]
X_es = [data["text"] for data in dataset_spanish]
y_es = [data["category"] for data in dataset_spanish]

X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X, y, test_size=0.1, random_state=42)
X_train_es, X_test_es, y_train_es, y_test_es = train_test_split(X_es, y_es, test_size=0.1, random_state=42)

ys = [y_train_en, y_test_en, y_train_es, y_test_es]

with open('ys.pkl', 'wb') as f:
        pickle.dump(ys, f)

vectorizers = {}
models = {}
datas = [X_train_en, X_test_en, X_train_es, X_test_es]

print("Data split into 90% train - 10% test\n")

def create_vecs(vectorizer):
    vecs = []
    i = 5
    for X in datas:
        if i%2 == 0:
            vecs.append(vectorizer.transform(preprocess(X)))
        else:
            vecs.append(vectorizer.fit_transform(preprocess(X)))
        i += 1
    return vecs

def save():
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(vectorizers, f)

print("Starting load Word2Vec")

### WORD2VEC

def get_sentence_rep(text, model, dim=300):
    words = text.lower().split()
    zero_vec = np.zeros(dim)
    avg_vec = np.zeros(dim)
    total_w = 0
    for w in words:
        try: 
            avg_vec += model.get_vector(w)
            total_w += 1
        except:
            pass   
    if total_w == 0:
        return zero_vec
    return avg_vec/total_w

vecs = []
model_word2vec_en = api.load('word2vec-google-news-300')
print("English model loaded")
vecs.append([get_sentence_rep(text, model_word2vec_en) for text in preprocess(datas[0])])
vecs.append([get_sentence_rep(text, model_word2vec_en) for text in preprocess(datas[1])])
print()
model_word2vec_es = KeyedVectors.load_word2vec_format('pre-trained-spanish/68/model.bin', binary=True, limit=None) 
print("Spanish model loaded")
vecs.append([get_sentence_rep(text, model_word2vec_es, dim=100) for text in preprocess(datas[2])])
vecs.append([get_sentence_rep(text, model_word2vec_es, dim=100) for text in preprocess(datas[3])])
print()
vectorizers["Word2Vec"] = vecs

print("Word2Vec loaded")

save()

print("Starting load BERT")

### BERT

def get_sentence_rep2(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vector = outputs[0][:, 0, :]  # Get the CLS vector
    return cls_vector

token_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')
token_bert_es = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
model_bert_es = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

vecs = []
print("English vectorization")
vecs.append([get_sentence_rep2(text, model_bert, token_bert) for text in datas[0]])
vecs.append([get_sentence_rep2(text, model_bert, token_bert) for text in datas[1]])
print("COMPLETED")

print("Spanish vectorization")
vecs.append([get_sentence_rep2(text, model_bert_es, token_bert_es) for text in datas[2]])
vecs.append([get_sentence_rep2(text, model_bert_es, token_bert_es) for text in datas[3]])
print("COMPLETED")
vectorizers["BERT"] = vecs

print("BERT loaded")

save()

print("Starting load RoBERTa")

import torch
from transformers import RobertaTokenizer, RobertaModel

### ROBERTA

token_roberta = RobertaTokenizer.from_pretrained('roberta-base')
model_roberta = RobertaModel.from_pretrained('roberta-base')
token_roberta_es = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model_roberta_es = RobertaModel.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')

vecs = []
print("English vectorization")
vecs.append([get_sentence_rep2(text, model_roberta, token_roberta) for text in datas[0]])
vecs.append([get_sentence_rep2(text, model_roberta, token_roberta) for text in datas[1]])
print("COMPLETED")

print("Spanish vectorization")
vecs.append([get_sentence_rep2(text, model_roberta_es, token_roberta_es) for text in datas[2]])
vecs.append([get_sentence_rep2(text, model_roberta_es, token_roberta_es) for text in datas[3]])
print("COMPLETED")
vectorizers["RoBERTa"] = vecs

print("RoBERTa loaded")

save()

### BINARY BAG OF WORD

vectorizer_bbw = CountVectorizer(binary=True)
vectorizers["Binary Bag of Words"] = create_vecs(vectorizer_bbw)

print("Binary Bag of Words loaded")

save()

from sklearn.feature_extraction.text import CountVectorizer

### BAG OF WORDS

vectorizer_bw = CountVectorizer()
vectorizers["Bag of Words"] = create_vecs(vectorizer_bw)

print("Bag of Words loaded")

save()

### BIGRAMS OF WORDS

vectorizer_biw = CountVectorizer(ngram_range=(2,2))
vectorizers["Bigrams of Words"] = create_vecs(vectorizer_biw)

print("Bigrams of Words loaded")

save()

### TRIGRAMS OF WORDS

vectorizer_triw = CountVectorizer(ngram_range=(3,3))
vectorizers["Trigrams of Words"] = create_vecs(vectorizer_triw)

print("Trigrams of Words loaded")

save()

### BIGRAMS OF CHARACTERS

vectorizer_bic = CountVectorizer(analyzer="char_wb", ngram_range=(2,2))
vectorizers["Bigrams of Characters"] = create_vecs(vectorizer_bic)

print("Bigrams of Characters loaded")

save()

### TRIGRAMS OF CHARACTERS

vectorizer_tric = CountVectorizer(analyzer="char_wb", ngram_range=(3,3))
vectorizers["Trigrams of Characters"] = create_vecs(vectorizer_tric)

print("Trigrams of Characters loaded")

save()

### TF-IDF

vectorizer_tfidf = TfidfVectorizer()
vectorizers["TF-IDF"] = create_vecs(vectorizer_tfidf)

print("TF-IDF loaded")

save()

### TF-IDF Binary

vectorizer_tfidfb = TfidfVectorizer(binary=True)
vectorizers["TF-IDF Binary"] = create_vecs(vectorizer_tfidfb)

print("TF-IDF Binary loaded")

save()

### IF-IDF Trigrams of Words

vectorizer_tfidf_biw = TfidfVectorizer(ngram_range=(2,2))
vectorizers["TF-IDF Bigrams of Words"] = create_vecs(vectorizer_tfidf_biw)

print("Bigrams of Words loaded")

save()

### TF-IDF Trigrams of Words

vectorizer_tfidf_triw = TfidfVectorizer(ngram_range=(3,3))
vectorizers["TF-IDF Trigrams of Words"] = create_vecs(vectorizer_tfidf_triw)

print("TF-IDF Trigrams of Words loaded")

save()

### TF-IDF Bigrams of Characters

vectorizer_tfidf_bic = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,2))
vectorizers["TF-IDF Bigrams of Characters"] = create_vecs(vectorizer_tfidf_bic)

print("TF-IDF Bigrams of Characters loaded")

save()

### TF-IDF Trigrams of Characters

vectorizer_tfidf_tric = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3))
vectorizers["TF-IDF Trigrams of Characters"] = create_vecs(vectorizer_tfidf_tric)

print("TF-IDF Trigrams of Characters loaded")

save()