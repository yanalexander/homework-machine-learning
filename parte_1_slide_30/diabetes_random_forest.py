import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("files/diabetes.csv")

# Divide os dados em dois conjuntos: Atributos e Classes
attributes = data.drop('class', axis=1)
classes = data['class']

# Dividir os dados aleatoriamente em conjunto para aprendizado e conjunto para testes
X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20) #20%

#Treinar o modelo
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

#Aplicar o modelo gerado sobre os dados separados para testes
y_pred = classifier.predict(X_test)

#Avaliar o modelo: Acurácia e matriz de contingência
print("Resultado da Avaliação do Modelo")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Classificar uma nova instância
nova_instancia=[[6,148,72,35,0,33.6,0.627,50]]
print(classifier.predict(nova_instancia))

#Salvar o modelo para uso posterior
joblib.dump(classifier, 'models/diabetes_random_forest.joblib')




