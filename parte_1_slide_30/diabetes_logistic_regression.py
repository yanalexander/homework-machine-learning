import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import joblib as jl
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

diabetes = pd.read_csv("files/diabetes.csv")

# Divide os dados em dois conjuntos: Atributos e Classes
attributes = diabetes.drop('class', axis=1)
classes = diabetes['class']

# Divide aleatoriamentes os conjuntos em teste e treino
X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)

# Criar e treinar modelo de regressão
logreg = LogisticRegression(solver='liblinear')
classifier = RFE(logreg, 20)
classifier = classifier.fit(X_train, y_train)

jl.dump(classifier,'models/logistic_regression_diabetes.joblib')

scores = cross_val_score(classifier,X_test,y_test,cv=30)

print('cross_val_score')
print(scores)
print('Precisao media: ',scores.mean())

scores_ = cross_validate(classifier,X_test,y_test,cv=30)
print('cross_validate')
print(scores_)
print('Precisao media:', scores_['test_score'].mean())

y_pred = classifier.predict(X_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# predict probabilities
probs = classifier.predict_proba(X_test)
print(probs)