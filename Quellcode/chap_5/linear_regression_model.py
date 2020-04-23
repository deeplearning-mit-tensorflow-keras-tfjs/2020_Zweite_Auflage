#
# Einfache lineare Regression mit TensorFlow 2.x
#
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

np.random.seed(42) # Damit wir immer die gleichen Zufallswerte bekommen
my_weight = 4 # Diese Variable muss später vom Modell gelernt werden.

# Ein Array von 100 Werten wird generiert.
input = np.arange(0, 10, 0.1,dtype="float32")
noise = np.random.uniform(-1,1,size=input.shape)

# Damit die Ausgabe nicht direkt linear ist, werden Zufallswerte hinzugefügt.
output = my_weight * input + noise
output = output.astype("float32")

plt.title('Funktion y=x*w + b')
plt.scatter(input, output, c="red")
plt.show()

class MyLinearRegressionModel():
    def __init__(self):
        self.W = tf.Variable(np.random.uniform(),dtype="float32",trainable=True )
        self.b = tf.Variable(np.random.uniform(),dtype="float32",trainable=True )

    def __call__(self, x):
        return tf.add(tf.multiply(x,self.W), self.b)

model = MyLinearRegressionModel()

def loss_function(pred, y):
    return tf.reduce_mean(tf.square(pred - y))

learning_rate=0.01
optimizer = tf.keras.optimizers.SGD(lr=learning_rate)

 # 1. Version
def train(model,x,y):
    with tf.GradientTape() as tape:
        current_loss = loss_function(model(input), output)  
    
    dW, db  = tape.gradient(current_loss, [model.W,model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db )
    return current_loss

 # 2. Version mit tape.gradient() und optimizer
def train_with_optimizers(model,x,y):
    with tf.GradientTape() as tape:
        current_loss = loss_function(model(input), output)  
    gradients = tape.gradient(current_loss,[model.W,model.b])
    optimizer.apply_gradients(zip(gradients , [model.W,model.b]))
    return current_loss


plt.ion()
plt.title('Funktion')
plt.legend(loc='upper left')

 # Update des plt.scatter() mit den neu berechneteten Werten
def redraw_curve(input,output,predicted_output,epoch,loss):
    plt.clf()
    plt.title("Epoch: " +str(epoch) +'\nLoss: ' + str(loss.numpy())+'\nFunktion y = x*w + b  mit w=' + str(model.W.numpy()) + ' und b=' + str(model.b.numpy()))
    plt.scatter(input,output,c="red",s=4,label="Original Werte")
    plt.scatter(input,predicted_output,s=5, c="g", label="Vorhersage")
    plt.show()  
    plt.draw()
    plt.pause(0.0001)

# Trainingsschleife
for epoch in range(0,500):
    # Version 1:
    loss = train_with_optimizers(model, input,output)
    # Version 2:
    # loss = train(model,input,output)
    print("Current loss: {}".format(loss.numpy()))
    predicted_output = model(input) 
    redraw_curve(input,output,predicted_output,epoch,loss)
   
plt.show(block=True) 
print("Vorhersage Wert für w: " + str(model.W.numpy()))