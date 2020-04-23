#
# Beispiel einer linearen Regression mit scikit-learn
#

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

# um die Reproduzierbarkeit der Ergebnisse zu gewährleisten, setzen wir random.seed auf eine festen Wert, z.B. 7n
np.random.seed(7)

# Der Array für die x-Werte des Datensatzes weren generiert
x = np.arange(-500,500)

# Der zu addierende Zufallswert wird in einem Streifen von ±15 um die Werte der Funktion y = .5 * x + 11 
# und die y-Werte des Datensatzes werden generiert.
delta = np.random.uniform(-15 , 15, x.size)
y = .5 * x + 11 + delta
print("X === ", x)
np.set_printoptions(precision=3)
print("RES == ", y)

# Das Dataset wird im Verhältnis von 80:20 in ein Trainingset 
# und ein Teset aufgeteilt
x_train = x[:-200].reshape( (800, 1) )
x_test = x[-200:].reshape( (200, 1) )
y_train = y[:-200]
y_test = y[-200:]

# Ein Instanz der Klasse für lineare Regression wird erzeugt
model = linear_model.LinearRegression()

# Das Modell wird mit den Trainsdaten (800 Einträgen) trainiert ... 
model.fit(x_train, y_train)

# ... und die Koeffizienten der linearen Regression ausgegeben.
print('Coefficient: \n', model.coef_)

# Das trainierte Modell wird nun mit den Testdaten getestet. 
# Die zu den x-Werten des Testsets korrespondierenden Werte werden 
# mit dem Modell berechnet.
y_pred = model.predict(x_test)

# Die Metriken zu den testwerten können nun ausgegeben werden.
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('r_2 statistic: %.2f' % r2_score(y_test, y_pred))

# Die Ergebnisse werden visualisiert. 
# durch Ploten des Testsets
plt.scatter(x_test, y_test)

# und Ploten der durch lienare Regression gelerneten Funktion
plt.plot(x_test, y_pred, color='red')
plt.show()
