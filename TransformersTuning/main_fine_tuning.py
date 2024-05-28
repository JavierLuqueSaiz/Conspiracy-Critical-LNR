#Import the training data for English and Spanish and
import pandas as pd
from datareader import en_train_df, es_train_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split
from fine_tuning import training, validate
SEED=1234
from utils import set_seed

if __name__ == "__main__":

    set_seed(SEED)

    lang="english"
    model_name="roberta-base"
    if lang == "spanish":
        X= es_train_df
    elif lang == "english":
        X= en_train_df
    else:
        X = pd.concat([en_train_df, es_train_df])


    #TODO add constraint to language and model
    if lang == "english":
        assert (model_name in ['bert-base-uncased', "roberta-base", 'microsoft/deberta-base'])
    if lang == "spanish":
        assert (model_name in ['dccuchile/bert-base-spanish-wwm-uncased','PlanTL-GOB-ES/roberta-base-bne'])
    if lang == "multi":
        assert (model_name in ['bert-base-multilingual-uncased'])

    # HPERPARAMETERS
    # 1 optimizer
    lr_scheduler, optimizer = None, None
    optimizer_name = "adam" #['adam', 'rmsprop', 'sgd']
    # 2 learning rate
    learning = 1e-5 #[1e-4, 0.5e-5, 1e-5, 0.5e-6, 1e-6]
    # 3 epochs
    epochs = 5 #[5, 10, 20]
    # 4 batch size
    batch_size = 32 #[16, 32, 64, 128]
    # 5 learning_rate schedule
    schedule = "linear" #['linear', 'cosine', constant]
    # 6 Quality measure name
    #measure = "Yeshwant123/mcc"
    measure = "mcc"

    patience = 3 #[5, 10]
    max_length = 128 #[This value can be estimated on the training set]


    print("Loading Tokenizer", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Transformer Model", model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    #Split training data in train and validation partition using hold out. It would be interesting use a K-Fold validation strategy.
    X_train, X_val = train_test_split(X, test_size=0.1, random_state=SEED, shuffle=True, stratify=X['label'])

    #FINE-TUNNING the model and obtaining the best model across all epochs
    fineTmodel=training(_model=model, _train_data=X_train, _val_data=X_val,_learning_rate= learning,
                        _optimizer_name=optimizer_name, _schedule=schedule,  _epochs=epochs, _tokenizer=tokenizer, _batch_size=batch_size,
                        _padding="max_length", _max_length=max_length, _truncation=True, _patience=patience, _measure= measure, _out="./out")


    #VALIDATING OR PREDICTIONG on the test partition, this time I'm using the validation set, but you have to use the test set.
    test_data=X_val
    preds=validate(_model=fineTmodel, _test_data=X_val, _tokenizer=tokenizer, _batch_size=batch_size, _padding="max_length", _max_length=max_length, _truncation=True, _measure=measure, evaltype=True)


    #SAVING THE PREDICTION IN THE PROPER FORMAT
    ' '





