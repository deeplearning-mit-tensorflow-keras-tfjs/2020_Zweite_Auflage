# Benutzung von Keras (in TensorFlow)
try:
    import tensorflow as tf
    from tensorflow import keras
    print("Keras TensorFlow version: {}".format(keras.__version__))
except:
    print("Keras TensorFlow Version nicht installiert")

# Wenn Sie zusätzlich Keras über keras.io installiert haben, werden Sie
# folgende Zeilen ausführen können
# Wenn es nicht der Fall sein sollte, 
# werden Sie folgende Fehlermeldung erhalten:
# ModuleNotFoundError: No module named 'keras' 
try:
    import keras
    print("Keras version: {}".format(keras.__version__))
except: 
    print("Keras Version von keras.io nicht installiert")