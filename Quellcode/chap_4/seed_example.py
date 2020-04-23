import numpy as np

# Es wird ein Vektor arr aus vier zufälligen Zahlen zwischen 0 und 1 generiert
arr = np.random.rand(4)
print(arr)

# arr hat wieder andere Werte
arr = np.random.rand(4)
print(arr)

# Jetzt wird der Zufallsgenerator mit einem festen Wert seed initialisiert
np.random.seed(23)

# arr hat einen zufälligen Wert, der aber reproduzierbar ist
arr = np.random.rand(4)
print(arr)

# Jetzt wird noch mal mit demselben seed gearbeitet ... 
np.random.seed(23)

# ... und arr hat denselben Wert wie eben.
arr = np.random.rand(4)
print(arr)


