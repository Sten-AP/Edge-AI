import os
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

BASE_DIR = os.path.dirname(__file__)

cifar10 = keras.datasets.cifar10
(train_img, train_labels), (test_img, test_labels) = cifar10.load_data( )

print(train_img.shape) 

train_img, test_img = train_img / 255.0, test_img / 255.0

ctg_names = [ "vliegtuig", "auto", "vogel", "kat", "hert", "hond", "kikker", "paard", "schip", "vrachtwagen" ]

def toon9():
	plt.figure(figsize=(4,4))
	for i in range(9):
		plt.subplot(3,3,i+1)
		plt.xticks( [ ] )
		plt.yticks( [ ] )
		plt.grid(False)
		plt.imshow( train_img[ i ], cmap=plt.cm.binary )
		plt.xlabel( ctg_names [ train_labels[ i ] [0] ] )
	plt.show()

model = keras.models.Sequential( ) 
filtersL=32 
kernelsL=(3,3)
poolsizeL=(2,2)

model.add(layers.Conv2D(filtersL, kernelsL, activation="relu", input_shape=(32, 32, 3))) 
model.add(layers.MaxPool2D(poolsizeL)) 
model.add(layers.Conv2D(filtersL, kernelsL, activation="relu"))
model.add(layers.MaxPool2D(poolsizeL))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
metrics = ["accuracy"]

model.compile(optimizer=optimizer , loss=loss, metrics=metrics)

epochs=10
batchsize=64

model.fit(train_img, train_labels, epochs=epochs, batch_size=batchsize)
model.evaluate(test_img, test_labels, batch_size=batchsize)
model.save(f"{BASE_DIR}/CNN_1")