import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import joblib as jl
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

artrodese = pd.read_csv("files/Ortopedia_Coluna.csv")

#Isolar a base de dados com a base minoritaria
minoritaria = artrodese[artrodese['fusao']==1]
majoritaria = artrodese[artrodese['fusao']==0]
minoritaria_upsample = resample(minoritaria,replace=True,n_samples=7900,random_state=123)

#Merge dataframes
attributes_balance = pd.concat([majoritaria,minoritaria_upsample])

classes = attributes_balance['fusao']

# Divide os dados em dois conjuntos: Atributos e Classes
attributes = attributes_balance.drop('fusao' , axis=1)
print(attributes)

#Cria dicionário e mapa para sexo
d = {'F': 0, 'M': 1}
attributes['Sexo'] = attributes['Sexo'].map(d).astype(int)

attributes = pd.get_dummies(attributes)
print(attributes)

# Divide aleatoriamentes os conjuntos em teste e treino
X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)

# Criar e treinar modelo de regressão
logreg = LogisticRegression(solver='liblinear')
classifier = RFE(logreg, 20)
classifier = classifier.fit(X_train, y_train)

jl.dump(classifier,'models/diabetes_logistic_regression.joblib')

y_pred = classifier.predict(X_test)
print(y_pred)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# predict probabilities
probs = classifier.predict_proba(X_test)
print(probs)