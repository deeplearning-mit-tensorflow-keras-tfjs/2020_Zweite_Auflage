import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
module = hub.Module("https://tfhub.dev/metmuseum/vision/classifier/imet_attributes_V1/1")
print(module)
model = tf.keras.models.load_model("met")
print(model)
#model.summary()
