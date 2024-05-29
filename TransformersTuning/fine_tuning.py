from mydataset import  MyDataset
from utils import remove_previous_model
import os
import torch
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
# TRAINING LOOP
from torch.optim import Adam, RMSprop
from transformers  import  get_scheduler

def training(_wandb, _model, _train_data, _val_data, _learning_rate, _optimizer_name, _schedule, _epochs,
             _tokenizer, _batch_size=32, _padding="max_length", _max_length=512, _truncation=True,
             _patience=10, _measure= "accuracy", _out=None):
    train_encodings = _tokenizer(_train_data["text"].tolist(), max_length=_max_length, truncation=_truncation, padding=_padding, return_tensors="pt")
    val_encodings = _tokenizer(_val_data["text"].tolist(), max_length=_max_length, truncation=_truncation, padding=_padding, return_tensors="pt")

    train, val = MyDataset(train_encodings, _train_data["label"].tolist()), MyDataset(val_encodings, _val_data["label"].tolist())

    train_dataloader, val_dataloader  = torch.utils.data.DataLoader(train, batch_size=_batch_size, shuffle=True), torch.utils.data.DataLoader(val, batch_size=_batch_size)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("The device available is:", str(device))
    _wandb.log({"device": str(device)})
    if use_cuda:
        model = _model.to(device)
          if torch.cuda.device_count() > 1:
              print(f"Usando {torch.cuda.device_count()} GPUs")
              model = torch.nn.DataParallel(model)
    best_measure, best_model_name, patience = None, None, 0
    training_stats = []
    # train_eval = evaluate.load("accuracy")
    train_eval = evaluate.load(f"Yeshwant123/{_measure}")

    _wandb.log({"info": "Creating the Optimizer and Schedule "})
    
    lr_scheduler, optimizer = None, None
    #Here we can specify different methods to optmize the paarameters, initially we can consider Adam and RmsProp

    print("Creating the Optimizer and Schedule")
    lr_scheduler, optimizer = None, None
    if _optimizer_name == "adam":
        optimizer = Adam(_model.parameters(), lr=_learning_rate)
    elif _optimizer_name == "rmsprop":
        optimizer = RMSprop(_model.parameters(), lr=_learning_rate)

    #Here we can define different learning rate schedules, to variate de learning rate in each training step. Initially we use
    # can use linear learning rate schedule
    num_training_steps = _epochs * len(_train_data)
    if _schedule=="linear":
        lr_scheduler = get_scheduler(_schedule, optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    for epoch in range(_epochs):
        if patience >= _patience: break
        total_loss_train, total_acc_train = 0, 0
        total_train_step = 0
        for batch in train_dataloader:
            total_train_step += 1
            # print("Epoch ", epoch, "Batch", i)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss_train += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            train_eval.add_batch(predictions=predictions, references=batch["labels"])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        total_acc_train = train_eval.compute()

        total_eval_steps = 0
        total_loss_val, total_acc_val = 0, 0
        eval_metric = evaluate.load(f"Yeshwant123/{_measure}")
        model.eval()
        for batch in val_dataloader:
            total_eval_steps += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                total_loss_val += loss.item()
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
            eval_metric.add_batch(predictions=predictions, references=batch["labels"])
        total_acc_val = eval_metric.compute()

        if best_measure is None or (best_measure < total_acc_val[_measure]):  # here you must set your save weights
            if best_measure == None:
              print("It's the first time (epoch) ******************")
              _wandb.log({"info": "It's the first time (epoch) ******************"})
            elif best_measure < total_acc_val[_measure]:
                print("In this epoch an improvement was achieved. (epoch) ******************")
                _wandb.log({"info": "In this epoch an improvement was achieved. (epoch) ******************"})
            best_measure = total_acc_val[_measure]
            try:
                os.makedirs(_out + os.sep + 'models', exist_ok=True)
            except OSError as error:
                print("Directory '%s' can not be created")
                _wandb.log({"info": "Directory '%s' can not be created"})
            # remove the directories
            remove_previous_model(_out + os.sep + 'models')
            best_model_name = _out + os.sep + 'models/bestmodel_epoch_{}'.format(epoch +1)
            print("The current best model is", best_model_name, best_measure)
            _wandb.log({f"info": "The current best model is {best_model_name} {best_measure}"})
            
            os.makedirs(best_model_name, exist_ok=True)
            model.save_pretrained(best_model_name)
            patience = 0
        else:
            patience += 1
        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': total_loss_train / total_train_step,
                'Valid. Loss': total_loss_val / total_eval_steps,
                f'Valid.{_measure}': total_acc_val[_measure],
                f'Training.{_measure}': total_acc_train[_measure]
            }
        )
        print(
            f"""Epochs: {epoch + 1} | Train Loss: {total_loss_train / total_train_step:.3f} | Train {_measure}: {total_acc_train[_measure]:.3f} | Val Loss: {total_loss_val / total_eval_steps:.3f} | Val {_measure}: {total_acc_val[_measure]:.3f}""")

        _wandb.log({
            'epoch': epoch + 1,
            'train_loss': total_loss_train / len(train_dataloader),
            f'train_{_measure}': total_acc_train[_measure],
            'val_loss': total_loss_val / len(val_dataloader),
            f'val_{_measure}': total_acc_val[_measure]
        })

    if best_model_name != None:
        model = model.from_pretrained(best_model_name)
        print("The final model used to predict the labels of the testing datasets is", best_model_name)
        _wandb.log({"info": f"The final model used to predict the labels of the testing datasets is {best_model_name}"})

    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    df_stats.to_csv(_out + os.sep + "training_stats.csv")

    _wandb.log({"info": df_stats})
    print(df_stats)
    myplot = sns.lineplot(data=df_stats, palette="tab10", linewidth=2.5)
    fig = myplot.get_figure()
    fig.savefig(_out + os.sep + 'loss-figue.png')
    plt.close()
    return model

############################################################################################################################################################################3
#VALIDATION ON THE TEST SET

def validate(_wandb, _model, _test_data, _tokenizer, _batch_size=32, _padding="max_length", _max_length=512, _truncation=True, _measure="accuracy", evaltype=True):
    test_encodings = _tokenizer(_test_data['text'].tolist(), max_length=_max_length, truncation=_truncation, padding=_padding, return_tensors="pt")
    _mode = "train" if evaltype else "test"
    test = MyDataset(test_encodings, _test_data["label"].tolist(), mode=_mode)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=_batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = _model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"Usando {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
    eval_metric, out, k = None, None, 0
    if evaltype==True:
        eval_metric = evaluate.load(f"Yeshwant123/{_measure}")

    model.eval()
    for batch in test_dataloader:

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            if k == 0:
                out = predictions
            else:
                out = torch.cat((out, predictions), 0)
            k += 1
            if evaltype==True:
                eval_metric.add_batch(predictions=predictions, references=batch["labels"])
    if evaltype==True:
        total_acc_test = eval_metric.compute()
        test_mesure = total_acc_test[_measure]
        avg_test_loss = total_loss / len(test_dataloader)        # Log the test accuracy and loss to wandb
        _wandb.log({
            f'test_{_measure}': test_mesure,
            'test_avg_loss': avg_test_loss
        })
    return out


