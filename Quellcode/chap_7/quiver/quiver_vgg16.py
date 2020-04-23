#
# Einsatz von Quiver zur interaktiven Visualisierung eines VGG16 Modells
#

from quiver_engine import server
from keras.applications import VGG16
global model

model = VGG16(weights="imagenet")
model.summary()

# Struktur des Modells
print(model.to_json())

# Quiverboard wird gestartet
server.launch(model,top=10,temp_folder="./tmp", input_folder='./imgs',port=12345)