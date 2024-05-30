#Import the training data for English and Spanish and
import pandas as pd
from datareader import en_train_df, es_train_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split, KFold
from fine_tuning import training, validate
SEED=1234
from utils import set_seed, product_dict
import wandb
from datetime import datetime

if __name__ == "__main__":

    # Get current date and time
    current_datetime = datetime.now()
    
    # Format it to include hours, minutes, and seconds
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    set_seed(SEED)

    preconfig = {
        #0: {
        #    "lang": "english",
        #    "model_name": "roberta-base",
        #},
        1: {
             "lang": "spanish",
             "model_name": "bert-base-multilingual-uncased"
         },
        2: {
            "lang": "english",
            "model_name": "microsoft/deberta-base",
        },
         3: {
             "lang": "spanish",
             "model_name": "dccuchile/bert-base-spanish-wwm-uncased",
         },
         4: {
             "lang": "spanish",
             "model_name": "PlanTL-GOB-ES/roberta-base-bne",
         }
    }
    
    hyperparams = {
        "optimizer_name": ["adam", "rmsprop"], # ["adam", "rmsprop", "sgd"]
        "learning": [1e-5], # [0.5e-5, 1e-5, 0.5e-6, 1e-6
        "schedule": ["linear", "cosine"], # ["linear", "cosine", "constant"]
        "patience": [5], # [3, 5, 10]
        "epochs": [10], # [5, 10, 20]
        "measure": ["mcc"],
        "batch_size": [32], # [16, 32, 64, 128]
        "max_length": [128]
    }

    kf = KFold(n_splits=5)

    for i, preconfig in preconfig.items():
        lang=preconfig["lang"]
        model_name=preconfig["model_name"]
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
            assert (model_name in ['dccuchile/bert-base-spanish-wwm-uncased','PlanTL-GOB-ES/roberta-base-bne','bert-base-multilingual-uncased'])
    
        # HPERPARAMETERS
        # 1 optimizer
        #lr_scheduler, optimizer = None, None
        #optimizer_name = "adam" #['adam', 'rmsprop', 'sgd']
        # 2 learning rate
        #learning = 1e-5 #[1e-4, 0.5e-5, 1e-5, 0.5e-6, 1e-6]
        # 3 epochs
        #epochs = 20 #[5, 10, 20]
        # 4 batch size
        #batch_size = 32 #[16, 32, 64, 128]
        # 5 learning_rate schedule
        #schedule = "linear" #['linear', 'cosine', constant]
        # 6 Quality measure name
        #measure = "Yeshwant123/mcc"
        #measure = "mcc"
    
        #patience = 3 #[5, 10]
        #max_length = 128 #[This value can be estimated on the training set]
    
        # Parent Run
        parent_run = wandb.init(project='LNR',
                                entity='javier-luque',
                                group=f'{lang}_{model_name}',
                                job_type='model')
        parent_run.config.update(preconfig)
        parent_run.config.update({"SEED":SEED})
        
        
        print("Loading Tokenizer", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)


        runs = 0

        for config in product_dict(**hyperparams):
            runs += 1

            # Start a child run for this hyperparameter configuration
            with wandb.init(project='LNR',
                            entity='javier-luque',
                            group=f'{lang}_{model_name}',
                            job_type='hyperparam-tuning',
                            name=f'{lang}_{model_name}_{runs}',
                            ) as run:
                # Log hyperparameters
                run.config.update(config)
                
                
            # For each fold
            
            #Split training data in train and validation partition using hold out. It would be interesting use a K-Fold validation strategy.
            #X_train, X_val = train_test_split(X, test_size=0.1, random_state=SEED, shuffle=True, stratify=X['label'])

            for fold, (train_index, val_index) in enumerate(kf.split(X)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                with wandb.init(project=f'LNR_{formatted_datetime}_FIN',
                                entity='javier-luque',
                                group=f'{lang}_{model_name}',
                                job_type=f'hyperparam-tuning-{runs}',
                                name=f'{lang}_{model_name}_{runs}_fold_{fold}'
                                ) as fold_run:
                    fold_run.config.update(config)
                    fold_run.config.update(config)
                    fold_run.config.update({"SEED":SEED})
    
                    # Log the fold number
                    fold_run.config.update({"fold": fold + 1})
                    print(f'Fold: {fold+1}')
    
                    wandb.log({"info": f"Loading Transformer Model {model_name}"})
                    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
                
                    #FINE-TUNNING the model and obtaining the best model across all epochs
                    fineTmodel=training(_wandb=fold_run, _model=model, _train_data=X_train, _val_data=X_val,_learning_rate=config["learning"],
                                        _optimizer_name=config["optimizer_name"], _schedule=config["schedule"],  _epochs=config["epochs"], _tokenizer=tokenizer, _batch_size=config["batch_size"],
                                        _padding="max_length", _max_length=config["max_length"], _truncation=True, _patience=config["patience"], _measure=config["measure"], _out="./out")
                   
                    #VALIDATING OR PREDICTIONG on the test partition, this time I'm using the validation set, but you have to use the test set.
                    test_data=X_val
                    preds=validate(_wandb=fold_run, _model=fineTmodel, _test_data=X_val, _tokenizer=tokenizer, _batch_size=config["batch_size"], _padding="max_length", _max_length=config["max_length"], _truncation=True, _measure=config["measure"], evaltype=True)
    
    
    
    
    
