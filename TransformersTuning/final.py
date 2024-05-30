import wandb
run = wandb.init()
artifacts = ['javier-luque/LNR_2024-05-30_14-03-15_FIN/run-n27mn2wt-history:v0']
for i in artifacts:
    artifact = run.use_artifact(i, type='wandb-history')
    artifact_dir = artifact.download()
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
