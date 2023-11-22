from keras.datasets import mnist
from extra_keras_datasets import emnist
import tensorflow as tf
import numpy


(x_train, y_train), (x_test, y_test) = emnist.load_data(type='letters')

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(27, activation=tf.nn.softmax))

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)

val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss:", val_loss, "Accuracy:", val_acc)

prediction = model.predict(x_train)
print(numpy.argmax(prediction[0]))
