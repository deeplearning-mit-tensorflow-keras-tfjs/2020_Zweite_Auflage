#
# Beispiele für die Benutzung von NumPy
# Bei Bedarf die Python-Module wie wget und tabulate mittels conda oder pip instalieren#
import numpy as np
import wget
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tabulate import tabulate
from scipy.stats import itemfreq

# Hier wird ein Datenset von Autos herunterladen
# https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

wget.download(DATA_URL,"auto.csv")
# usecols wird benutzt, um nur bestimmte Spalten zu laden
# make: 2
# fuel: 3
# num-of-doors: 5
# price: 25

dt = [("brand","S8"),("fuel","S10"),("doors","S10"),("price",float)]
all_autos= np.loadtxt("auto.csv",delimiter=",",dtype=str,usecols=[2,3,5,25])

# Alle Preise in einer Spalte
all_autos_prices = np.array(all_autos[:,3])

# Manche Zeilen besitzen einen ?. Wir ersetzen diese bevor wir diese Spalte in float umkonvertieren
all_autos_prices = np.core.defchararray.replace(np.array(all_autos[:,3]),'?','0').astype("float")
all_autos[:,3] = np.array(all_autos_prices,dtype="float32")


# Die Zeilen deren Preis gleich 0 ist, werden aus dem Datenset rausgenommen
no_prices = np.where(all_autos_prices == 0.0)
all_autos = np.delete(all_autos,no_prices,axis=0)
all_autos_prices = np.delete(all_autos_prices,no_prices,axis=0)

all_autos = np.array(all_autos.reshape(len(all_autos),4))

print("Durschnittspreis ${} ".format(np.mean(all_autos_prices))) # Das sind nur strings
print("Maximum ${} {}".format(np.max(all_autos_prices), all_autos[np.argmax(all_autos_prices)]))
print("Minimum ${} {}".format(np.min(all_autos_prices), all_autos[np.argmin(all_autos_prices)]))

print ("Umkonvertierung in Euro")
dollar_euro_conversion_rate = 0.88

# Wir selektieren die Spalte mit dem index 3 und konvertieren den Preis nach Euros
all_autos[:,3] = np.multiply(np.array(all_autos[:,3],dtype="float"),dollar_euro_conversion_rate)

print (tabulate(all_autos,headers=["Marke","Treibstoff","Anzahl Türe","Preis in Euro"]))

# Anzahl der VW im Datenset
nb_vw = len(np.where(all_autos[:,0] == "volkswagen")[0])
print("Anzahl von VW Autos: {}".format(nb_vw))

# Anzahl der Autos per Marke
# Wir benutzen hier itemfreq von scipy.stats
stats = np.array(itemfreq(all_autos[:,0]))

# Optional: Sortierung nach Anzahl der Fahrzeuge pro Marke 
#stats = stats[stats[:,1].astype(int).argsort()]

# Konvertierung zu einem record 
d = np.rec.fromarrays([stats[:,0],stats[:,1]],formats=['<U21','int'], names=['brand','numbers'])

# Sortiere nach Anzahl der Fahrzeuge pro Marke
d = d[d["numbers"].astype(int).argsort()]

# Erste Buchstabe der Automarke soll gross geschrieben werden
d["brand"] = np.char.capitalize(d["brand"])

# Ausgabe mit Matplotlib
plt.xticks(range(len(d)),d["brand"],rotation='vertical')
plt.bar(range(len(d)),d["numbers"])
plt.title("Anzahl der Fahrzeuge")
plt.xlabel("Marke")
plt.ylabel("Anzahl")
plt.show()
