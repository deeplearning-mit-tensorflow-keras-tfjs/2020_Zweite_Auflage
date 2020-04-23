#
# Beispiel der Klassifikations von Iris-Blumen
#

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt

# Lade den Iris-Datenset
data_train = pd.read_csv('./iris.csv')

# Die 3 zu erkennenden Klassifikationsklassen werden zu numerischen Werten 0, 1 bzw. 2 umgewandelt.
data_train.loc[data_train['species']=='Iris-setosa', 'species']=0
data_train.loc[data_train['species']=='Iris-versicolor', 'species']=1
data_train.loc[data_train['species']=='Iris-virginica', 'species']=2
data_train = data_train.apply(pd.to_numeric)

# Der eingelesene Datenset wird als Matrix dargestellt
data_train_array = data_train.values # oder data_train.to_numpy()

# Zur Sicherstellung der Reproduzierbarkeit der Ergebnisse setzen wir random.seed auf eine festen Wert, z.B. 42
np.random.seed(17)

# Das Datenset wird in zwei separate Kategorie gespaltet: Testdaten und Trainingsdaten. 
# 80% der Daten werden zum Trainieren und 20% zum Testen des Modells verwendet. 
# Da es sich bei der Eingabe um einen Vektor handelt, werden wird den Großbuchstaben X benutzen; 
# Für die Ausgabe hingegen handelt es sich um ein einzelner Werte, 
# daher die Bezeichung mit dem Kleinbuchstaben y 

X_train, X_test, y_train, y_test = train_test_split(data_train_array[:,:4],
                                                    data_train_array[:,4],
                                                    test_size=0.2)

# VERSION 1
# Ein neuronales Netz zur Klassifikation (MultiLayerPerceptron) wird mit folgenden Eigenschaften gebildet:
# einem Input-Layer mit 4 Neuronen, die die Merkmale der Iris-Planze repräsentieren;
# einem Hidden-Layer mit 10 Neuronen
# eime Output-Layer mit 4 Neuronen, die die zu erkennenden Klassen repräsentieren.
# Dabei wird als Aktivierungsfunktion relu und als Optimierer adam verwenden.
mlp = MLPClassifier(hidden_layer_sizes=(10,),activation='relu', solver='adam', max_iter=350, batch_size=10, verbose=True)

# VERSION 2
# In der zweiten Variante werden 2 Hidden-Layers mit jeweils 5 bzw. 3 Neuronen verwendet
# Dabei wird als Aktivierungsfunktion tanh und als Optimierer adam verwenden. 
#mlp = MLPClassifier(hidden_layer_sizes=(5,3),activation='tanh', solver='adam', max_iter=350, batch_size=10, verbose=True)

# Das neuronale Netz wird mit den Trainingsdaten traniert
mlp.fit(X_train, y_train)

# Das Ergebnis des Training wird ausgegeben
print("Trainingsergebnis: %5.3f" % mlp.score(X_train, y_train))

# Das Modell wird mit den Testdatensdaten evaluiert
predictions = mlp.predict(X_test)
# und die Konfusionsmatrix ausgegeben
print(confusion_matrix(y_test,predictions))  

# Aus der Konfusionsmatrix werden precison, recall und f1-score berechnet und ausgebenen
print(classification_report(y_test,predictions)) 

# Das Modell wird getest und das Ergebnis ausgegeben
print("Testergebnis: %5.3f" % mlp.score(X_test,y_test))

# Folgendes gibt die Werte der Gewichte pro Layer aus
print("WEIGHTS:", mlp.coefs_)
print("BIASES:", mlp.intercepts_) 

# Das Modell wird beispielsweise zur Vorhersage auf folgenden Werten 
# aus dem Testset angewandt mit den Merkmalen [sepal-length, sepal-width, 
# petal-length, petal-width]
print(mlp.predict([[5.1,3.5,1.4,0.2], [5.9,3.,5.1,1.8], [4.9,3.,1.4,0.2], [5.8,2.7,4.1,1.]]))

# Die Loss-Kurve wird visualisiert und in der Datei Plot_of_loss_values.png im PNG-Format gespeichert.
loss_values = mlp.loss_curve_
plt.plot(loss_values)
plt.savefig("./Plot_of_loss_values.png")
plt.show()
