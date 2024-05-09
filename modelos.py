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

print("#"*20)
print()

with open('ys.pkl', 'rb') as f:
    ys = pickle.load(f)

print("Dependent variable loaded")
print("-"*20)

with open('embeddings.pkl', 'rb') as f:
    vectorizations = pickle.load(f)

print("Embeddings loaded")
print("-"*20)

new_bert = []
for x in vectorizations["BERT"]:
    r = np.array(x).shape[0]
    c = np.array(x).shape[2]
    new_bert.append(np.array(x).reshape(r,c))
vectorizations["BERT"] = new_bert
new_roberta = []
for x in vectorizations["RoBERTa"]:
    r = np.array(x).shape[0]
    c = np.array(x).shape[2]
    new_roberta.append(np.array(x).reshape(r,c))
vectorizations["RoBERTa"] = new_roberta

print(f"Embeddings ({len(vectorizations)}):")

for i in vectorizations.keys():
    print(i)

print()
print("#"*20)
print()
print("MODELS")
print()

results_en = pd.DataFrame(index=vectorizations.keys())
results_es = pd.DataFrame(index=vectorizations.keys())

def save():
    with open('results2.pkl', 'wb') as f:
        results = (results_en, results_es)
        pickle.dump(results, f)

def svc(kernel, X, y = ys):
    X_train_en = X[0]
    X_test_en = X[1]
    X_train_es = X[2]
    X_test_es = X[3]
    y_train_en = y[0]
    y_test_en = y[1]
    y_train_es = y[2]
    y_test_es = y[3]

    # English

    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train_en, y_train_en)
    predicted = clf.predict(X_test_en)
    quality_en = matthews_corrcoef(y_test_en, predicted)
    print(f"SVC {kernel}. MCC for English: {quality_en}")

    # Spanish

    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train_es, y_train_es)
    predicted = clf.predict(X_test_es)
    quality_es = matthews_corrcoef(y_test_es, predicted)
    print(f"SVC {kernel}. MCC for Spanish: {quality_es}")

    return quality_en, quality_es

def lr(penalty, X, y=ys):
    X_train_en = X[0]
    X_test_en = X[1]
    X_train_es = X[2]
    X_test_es = X[3]
    y_train_en = y[0]
    y_test_en = y[1]
    y_train_es = y[2]
    y_test_es = y[3]

    # English

    clf = LogisticRegression(penalty=penalty)
    clf.fit(X_train_en, y_train_en)
    predicted = clf.predict(X_test_en)
    quality_en = matthews_corrcoef(y_test_en, predicted)
    print(f"Linear Regression {penalty} penalty. MCC for English: {quality_en}")

    # Spanish

    clf = LogisticRegression(penalty=penalty)
    clf.fit(X_train_es, y_train_es)
    predicted = clf.predict(X_test_es)
    quality_es = matthews_corrcoef(y_test_es, predicted)
    print(f"Linear Regression {penalty} penalty. MCC for Spanish: {quality_es}")

    return quality_en, quality_es

def dt(criterion, X, y = ys):
    X_train_en = X[0]
    X_test_en = X[1]
    X_train_es = X[2]
    X_test_es = X[3]
    y_train_en = y[0]
    y_test_en = y[1]
    y_train_es = y[2]
    y_test_es = y[3]

    # English

    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(X_train_en, y_train_en)
    predicted = clf.predict(X_test_en)
    quality_en = matthews_corrcoef(y_test_en, predicted)
    print(f"Decision Tree {criterion}. MCC for English: {quality_en}")

    # Spanish

    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(X_train_es, y_train_es)
    predicted = clf.predict(X_test_es)
    quality_es = matthews_corrcoef(y_test_es, predicted)
    print(f"Decision Tree {criterion}. MCC for Spanish: {quality_es}")

    return quality_en, quality_es

def mlp(solver, X, y=ys):
    X_train_en = X[0]
    X_test_en = X[1]
    X_train_es = X[2]
    X_test_es = X[3]
    y_train_en = y[0]
    y_test_en = y[1]
    y_train_es = y[2]
    y_test_es = y[3]

    # English

    clf = MLPClassifier(solver=solver, max_iter = 1300)
    clf.fit(X_train_en, y_train_en)
    predicted = clf.predict(X_test_en)
    quality_en = matthews_corrcoef(y_test_en, predicted)
    print(f"Multilayer Perceptron {solver} solver. MCC for English: {quality_en}")

    # Spanish

    clf = MLPClassifier(solver=solver, max_iter = 1300)
    clf.fit(X_train_es, y_train_es)
    predicted = clf.predict(X_test_es)
    quality_es = matthews_corrcoef(y_test_es, predicted)
    print(f"Multilayer Perceptron {solver} solver. MCC for Spanish: {quality_es}")

    return quality_en, quality_es


for name, vector in vectorizations.items():

    if name not in ["Word2Vec", "BERT", "RoBERTa"]:

        print(f"MODELS WITH {name} EMBEDDING")

        print("#"*20)
        print()
        print("Support Vector Machines (SVC)")
        print()

        print("Starting SVC linear\n")

        q_en, q_es = svc("linear", vector)
        results_en.loc[[name],["SVC linear"]] = q_en
        results_es.loc[[name],["SVC linear"]] = q_es
        save()

        print("-"*20)

        print("Starting SVC poly")

        q_en, q_es = svc("poly", vector)
        results_en.loc[[name],["SVC poly"]] = q_en
        results_es.loc[[name],["SVC poly"]] = q_es
        save()

        print("-"*20)

        print("Starting SVC rbf")

        q_en, q_es = svc("rbf", vector)
        results_en.loc[[name],["SVC rbf"]] = q_en
        results_es.loc[[name],["SVC rbf"]] = q_es
        save()

        print("-"*20)

        print("Starting SVC sigmoid")

        q_en, q_es = svc("sigmoid", vector)
        results_en.loc[[name],["SVC sigmoid"]] = q_en
        results_es.loc[[name],["SVC sigmoid"]] = q_es
        save()

        print("-"*20)

        print()
        print("#"*20)
        print()
        print("Logistic Regression (LR)")
        print()

        print("Starting Logistic Regression l2")

        q_en, q_es = lr("l2", vector)
        results_en.loc[[name],["LR l2"]] = q_en
        results_es.loc[[name],["LR l2"]] = q_es
        save()

        print("-"*20)

        print("Starting Logistic Regression None")

        q_en, q_es = lr(None, vector)
        results_en.loc[[name],["LR None"]] = q_en
        results_es.loc[[name],["LR None"]] = q_es
        save()

        print("-"*20)

        print()
        print("#"*20)
        print()
        print("Decision Trees (DTs)")
        print()

        print("Decision Tree gini")

        q_en, q_es = dt("gini", vector)
        results_en.loc[[name],["DTs gini"]] = q_en
        results_es.loc[[name],["DTs gini"]] = q_es
        save()

        print("-"*20)

        print("Decision Tree entropy")

        q_en, q_es = dt("entropy", vector)
        results_en.loc[[name],["DTs entropy"]] = q_en
        results_es.loc[[name],["DTs entropy"]] = q_es
        save()

        print("-"*20)

        print("Decision Tree log_loss")

        q_en, q_es = dt("log_loss", vector)
        results_en.loc[[name],["DTs log_loss"]] = q_en
        results_es.loc[[name],["DTs log_loss"]] = q_es
        save()

        print("-"*20)

        print()
        print("#"*20)
        print()
        print("Multilayer Perceptron (MLP)")
        print()

        print("Multilayer Perceptron lbfgs")

        q_en, q_es = mlp("lbfgs", vector)
        results_en.loc[[name],["MLP lbfgs"]] = q_en
        results_es.loc[[name],["MLP lbfgs"]] = q_es
        save()

        print("-"*20)

        print("Multilayer Perceptron sgd")

        q_en, q_es = mlp("sgd", vector)
        results_en.loc[[name],["MLP sgd"]] = q_en
        results_es.loc[[name],["MLP sgd"]] = q_es
        save()

        print("-"*20)

        print("Multilayer Perceptron adam")

        q_en, q_es = mlp("adam", vector)
        results_en.loc[[name],["MLP adam"]] = q_en
        results_es.loc[[name],["MLP adam"]] = q_es
        save()

        print("-"*20)

        print()
        print("#"*20)
        print()
        print("ENSEMBLES")
        print()

        best_SVC_en = results_en.loc[[name]].filter(like="SVC").idxmax(axis=1).iloc[0].split()[-1].lower()
        best_SVC_es = results_es.loc[[name]].filter(like="SVC").idxmax(axis=1).iloc[0].split()[-1].lower()

        best_LR_en = results_en.loc[[name]].filter(like="LR").idxmax(axis=1).iloc[0].split()[-1].lower()
        best_LR_es = results_es.loc[[name]].filter(like="LR").idxmax(axis=1).iloc[0].split()[-1].lower()

        best_DTs_en = results_en.loc[[name]].filter(like="DTs").idxmax(axis=1).iloc[0].split()[-1].lower()
        best_DTs_es = results_es.loc[[name]].filter(like="DTs").idxmax(axis=1).iloc[0].split()[-1].lower()

        best_MLP_en = results_en.loc[[name]].filter(like="MLP").idxmax(axis=1).iloc[0].split()[-1].lower()
        best_MLP_es = results_es.loc[[name]].filter(like="MLP").idxmax(axis=1).iloc[0].split()[-1].lower()

        print()
        print("#"*20)
        print()
        print("Voting hard")
        print()

        # English

        model1_en = svm.SVC(kernel=best_SVC_en)
        model2_en = LogisticRegression(penalty=best_LR_en)
        model3_en = DecisionTreeClassifier(criterion=best_DTs_en)
        ensemble_en = VotingClassifier(
            estimators=[('lr', model2_en),
                        ('dt', model3_en),
                        ('svm', model1_en)],
                        voting='hard')
        ensemble_en.fit(vector[0], ys[0])
        predicted = ensemble_en.predict(vector[1])
        quality_en = matthews_corrcoef(ys[1], predicted)
        results_en.loc[[name],["Voting hard"]] = quality_en
        print(f"Voting hard. MCC for English: {quality_en}")

        # Spanish

        model1_es = svm.SVC(kernel=best_SVC_es)
        model2_es = LogisticRegression(penalty=best_LR_es)
        model3_es = DecisionTreeClassifier(criterion=best_DTs_es)
        ensemble_es = VotingClassifier(
            estimators=[('lr', model2_es),
                        ('dt', model3_es),
                        ('svm', model1_es)],
                        voting='hard')
        ensemble_es.fit(vector[2], ys[2])
        predicted = ensemble_es.predict(vector[3])
        quality_es = matthews_corrcoef(ys[3], predicted)
        results_es.loc[[name],["Voting hard"]] = quality_es
        print(f"Voting hard. MCC for Spanish: {quality_es}")

        save()

        print()
        print("#"*20)
        print()
        print("Bagging")
        print()

        # English

        ensemble_en = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(criterion=best_DTs_en),
            n_estimators = 15
        )
        ensemble_en.fit(vector[0], ys[0])
        predicted = ensemble_en.predict(vector[1])
        quality_en = matthews_corrcoef(ys[1], predicted)
        results_en.loc[[name],["Bagging"]] = quality_en
        print(f"Bagging. MCC for English: {quality_en}")

        # Spanish

        ensemble_es = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(criterion=best_DTs_es),
            n_estimators = 15
        )
        ensemble_es.fit(vector[2], ys[2])
        predicted = ensemble_es.predict(vector[3])
        quality_es = matthews_corrcoef(ys[3], predicted)
        results_es.loc[[name],["Bagging"]] = quality_es
        print(f"Bagging. MCC for Spanish: {quality_es}")

        save()

        print()
        print("#"*20)
        print()
        print("Random Forest")
        print()

        # English

        ensemble_en = RandomForestClassifier(
            n_estimators = 130,
            max_depth = 15,
            random_state = 0
        )
        ensemble_en.fit(vector[0], ys[0])
        predicted = ensemble_en.predict(vector[1])
        quality_en = matthews_corrcoef(ys[1], predicted)
        results_en.loc[[name],["Random Forest"]] = quality_en
        print(f"Random Forest. MCC for English: {quality_en}")

        # Spanish

        ensemble_es = RandomForestClassifier(
            n_estimators = 130,
            max_depth = 15,
            random_state = 0
        )
        ensemble_es.fit(vector[2], ys[2])
        predicted = ensemble_es.predict(vector[3])
        quality_es = matthews_corrcoef(ys[3], predicted)
        results_es.loc[[name],["Random Forest"]] = quality_es
        print(f"Random Forest. MCC for Spanish: {quality_es}")

        save()

        print()
        print("#"*20)
        print()
        print("Boosting")
        print()

        # English

        ensemble_en = AdaBoostClassifier(
            base_estimator = LogisticRegression(penalty=best_LR_en),
            n_estimators = 15
        )
        ensemble_en.fit(vector[0], ys[0])
        predicted = ensemble_en.predict(vector[1])
        quality_en = matthews_corrcoef(ys[1], predicted)
        results_en.loc[[name],["Boosting"]] = quality_en
        print(f"Boosting. MCC for English: {quality_en}")

        # Spanish

        ensemble_es = AdaBoostClassifier(
            base_estimator = LogisticRegression(penalty=best_LR_es),
            n_estimators = 15
        )
        ensemble_es.fit(vector[2], ys[2])
        predicted = ensemble_es.predict(vector[3])
        quality_es = matthews_corrcoef(ys[3], predicted)
        results_es.loc[[name],["Boosting"]] = quality_es
        print(f"Boosting. MCC for Spanish: {quality_es}")

        save()

        print()
        print("#"*20)
        print()
        print("Stacking")
        print()

        # English

        model1_en = svm.SVC(kernel=best_SVC_en)
        model2_en = LogisticRegression(penalty=best_LR_en)
        model3_en = DecisionTreeClassifier(criterion=best_DTs_en)
        model4_en = KNeighborsClassifier()
        base_models_en = [('dt', model3_en),
                        ('svm', model1_en),
                        ('knn', model4_en)]
        ensemble_en = StackingClassifier(
            estimators = base_models_en,
            final_estimator = model2_en
        )
        ensemble_en.fit(vector[0], ys[0])
        predicted = ensemble_en.predict(vector[1])
        quality_en = matthews_corrcoef(ys[1], predicted)
        results_en.loc[[name],["Stacking"]] = quality_en
        print(f"Stacking. MCC for English: {quality_en}")

        # Spanish

        model1_es = svm.SVC(kernel=best_SVC_es)
        model2_es = LogisticRegression(penalty=best_LR_es)
        model3_es = DecisionTreeClassifier(criterion=best_DTs_es)
        model4_es = KNeighborsClassifier()
        base_models_es = [('dt', model3_es),
                        ('svm', model1_es),
                        ('knn', model4_es)]
        ensemble_es = StackingClassifier(
            estimators = base_models_es,
            final_estimator = model2_es
        )
        ensemble_es.fit(vector[2], ys[2])
        predicted = ensemble_es.predict(vector[3])
        quality_es = matthews_corrcoef(ys[3], predicted)
        results_es.loc[[name],["Stacking"]] = quality_es
        print(f"Stacking. MCC for Spanish: {quality_es}")

        save()

    





    






    


    



