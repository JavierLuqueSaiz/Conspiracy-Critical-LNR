#Import the training data for English and Spanish and
import pandas as pd
from datareader import en_train_df, es_train_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split
from fine_tuning_PEFT import training, validate
from peft import LoraConfig, get_peft_model, TaskType


if __name__ == "__main__":
    lang="english"
    model_name= "bert-base-uncased"
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
    optimizer_name = "adam"
    # 2 learning rate
    learning = 1e-5
    # 3 epochs
    epochs = 20
    # 4 batch size
    batch_size = 32
    # 5 learning_rate schedule
    schedule = "linear"
    # 6 Quality measure name
    #measure = "Yeshwant123/mcc"
    measure = "mcc"
    patience = 10
    max_length = 128



    print("Loading Tokenizer", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Transformer Model", model_name)
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    #INCLUEDE THE PEFT ESTRATEGY
    # Define LoRA Config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # this is necessary
        inference_mode=True
    )

    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()  # see % trainable parameters
    #print(dir(model))
    #print(model.model)

    #Split training data in train and validation partition using hold out. It would be interesting use a K-Fold validation strategy.
    X_train, X_val = train_test_split(X, test_size=0.1, random_state=1234, shuffle=True, stratify=X['label'])

    #FINE-TUNNING the model and obtaining the best model across all epochs
    fineTmodel=training(_model=model, _base=base, _train_data=X_train, _val_data=X_val,_learning_rate= learning,
                        _optimizer_name=optimizer_name, _schedule=schedule,  _epochs=epochs, _tokenizer=tokenizer, _batch_size=batch_size,
                        _padding="max_length", _max_length=max_length, _truncation=True, _patience=patience, _measure= measure, _out="./out")



    #merged_model.save_pretrained("merged-model")

    #VALIDATING OR PREDICTIONG on the test partition
    test_data=X_val
    preds=validate(_model=fineTmodel, _test_data=test_data, _tokenizer=tokenizer, _batch_size=batch_size, _padding="max_length", _max_length=max_length, _truncation=True, _measure=measure, evaltype=True)


    #SAVING THE PREDICTION IN THE PROPER FORMAT





