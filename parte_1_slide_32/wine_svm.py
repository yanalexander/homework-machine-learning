import pandas as pd
import joblib

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#data = pd.read_csv("files/winequality-red.csv", sep=";")
data = pd.read_csv("files/winequality-white.csv", sep=";")
#Cria dicionário e mapa para quality
d = {10: 'A', 9: 'A', 8: 'A', 7: 'B', 6: 'B', 5: 'B', 4: 'B', 3: 'C', 2: 'C', 1: 'C'}
data['quality'] = data['quality'].map(d)

# Divide os dados em dois conjuntos: Atributos e Classes
attributes = data.drop('quality', axis=1)
classes = data['quality']

# Dividir os dados aleatoriamente em conjunto para aprendizado e conjunto para testes
X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20) #20%

#Treinar o modelo
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

#Aplicar o modelo gerado sobre os dados separados para testes
y_pred = classifier.predict(X_test)

#Avaliar o modelo: Acurácia e matriz de contingência
print("Resultado da Avaliação do Modelo")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Classificar uma nova instância
nova_instancia=[[7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8]]
print(classifier.predict(nova_instancia))

#Salvar o modelo para uso posterior
joblib.dump(classifier, 'models/wine_svm.joblib')




