#
# Einfache lineare Regression mit TensorFlow
#

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

np.random.seed(42) # Damit wir immer die gleichen Zufallswerte bekommen
my_weight = 4 # Diese Variable muss später vom Modell gelernt werden.

# Ein Array von 100 Werten wird generiert.
input = np.arange(0, 10, 0.1)
noise = np.random.randint(low=-5, high=5, size=input.shape)

# Damit die Ausgabe nicht direkt linear ist, werden Zufallswerte hinzugefügt.
output = my_weight * input + noise
plt.scatter(input, output, c="red")
plt.show()

X = tf.placeholder(tf.float32,name="X") 
Y = tf.placeholder(tf.float32,name="Y")

w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

def linear_regression_model(X, w, b ):
    return tf.add(tf.multiply(X, w),b) 

# Kostenfunktion
cost = tf.square(Y - linear_regression_model(X, w, b)) 

# Trainings Operation
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 

epochs = range(0,100)

# Launch the graph in a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in epochs:
        print("--- Epoch {} ---".format(i))
        for (x, y) in zip(input,output):
            sess.run(train_op, feed_dict={X: x, Y: y})

        weight_value = sess.run(w)  # Gewichtungen und Biases werden berechnet 
        bias_value = sess.run(b)
        
        print("bias_value :{}".format(bias_value))
        print("weight_value :{} ".format(weight_value))
            
    predicted_output = sess.run(linear_regression_model(input,weight_value,bias_value))       
    
    # Ausgabe der Werte 
    plt.title('Funktion')
    plt.scatter(input,output,c="red",s=4,label="Original Werte")
    plt.scatter(input,predicted_output,s=5, c="g", label="Vorhersage")
    plt.legend(loc='upper left')
    plt.title('Funktion y = x*w + b  mit w=' + str(weight_value) + ' und b=' + str(bias_value))
    print("Vorhersage Wert für w: " + str(weight_value))
   
plt.show()
