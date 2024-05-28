import os
import wandb
import pandas as pd
import torch
import torch.distributed as dist
from datareader import en_train_df, es_train_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split
from fine_tuning_PEFT import training, validate
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn.parallel import DistributedDataParallel as DDP

def init_distributed_mode():
    """ Initialize distributed training """
    dist.init_process_group(backend='gloo', init_method='env://')
    torch.manual_seed(1234)

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="LNR", entity="javier-luque-saiz")

    # Initialize distributed mode
    init_distributed_mode()

    # Hyperparameters and settings
    lang = "english"
    model_name = "bert-base-uncased"
    if lang == "spanish":
        X = es_train_df
    elif lang == "english":
        X = en_train_df
    else:
        X = pd.concat([en_train_df, es_train_df])

    if lang == "english":
        assert (model_name in ['bert-base-uncased', "roberta-base", 'microsoft/deberta-base'])
    if lang == "spanish":
        assert (model_name in ['dccuchile/bert-base-spanish-wwm-uncased', 'PlanTL-GOB-ES/roberta-base-bne'])
    if lang == "multi":
        assert (model_name in ['bert-base-multilingual-uncased'])

    # Hyperparameters
    lr_scheduler, optimizer = None, None
    optimizer_name = "adam"
    learning = 1e-5
    epochs = 20
    batch_size = 32
    schedule = "linear"
    measure = "mcc"
    patience = 10
    max_length = 128

    print("Loading Tokenizer", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Transformer Model", model_name)
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Define LoRA Config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=True
    )

    model = get_peft_model(base, lora_config)
    model = DDP(model)  # Wrap model with DDP
    model.print_trainable_parameters()

    X_train, X_val = train_test_split(X, test_size=0.1, random_state=1234, shuffle=True, stratify=X['label'])

    fineTmodel = training(
        _model=model, _base=base, _train_data=X_train, _val_data=X_val, _learning_rate=learning,
        _optimizer_name=optimizer_name, _schedule=schedule, _epochs=epochs, _tokenizer=tokenizer, _batch_size=batch_size,
        _padding="max_length", _max_length=max_length, _truncation=True, _patience=patience, _measure=measure, _out="./out"
    )

    # Save the fine-tuned model to wandb
    wandb.save("./out/pytorch_model.bin")
    wandb.save("./out/config.json")
    wandb.save("./out/tokenizer_config.json")

    test_data = X_val
    preds = validate(
        _model=fineTmodel, _test_data=test_data, _tokenizer=tokenizer, _batch_size=batch_size,
        _padding="max_length", _max_length=max_length, _truncation=True, _measure=measure, evaltype=True
    )
