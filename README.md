# Conspiracy-Critical-LNR
Project on embeddings an supervised machine learning models for a classification task consisiting on identify a text as conspiracy or critical.

The order of execution is:
- First vectorization.py, which generates ys.pkl and embeddings.pkl.
- Then modelos.py, which genereates results2.pkl.

results2.pkl contains the dataframes for spanish and english MCC for each model and each embedding. There's and small visualization of these dataframes in seleccion.ipynb.

We are currently working on adding the pickle files on the repository. Because of computational costs the pickled were generated "piece by piece" and then added in the code, but we delivered the code to generate the files as a whole. The "whole" files will eventually be commited.
