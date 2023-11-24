import os
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

BASE_DIR = os.path.dirname(__file__)

cifar100 = keras.datasets.cifar100
(train_img, train_labels), (test_img, test_labels) = cifar100.load_data( )

print(train_img.shape) 

train_img, test_img = train_img / 255.0, test_img / 255.0

ctg_names = [  
'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard',
'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 
'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm' 
]

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
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(100))

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
metrics = ["accuracy"]

model.compile(optimizer=optimizer , loss=loss, metrics=metrics)

epochs=20
batchsize=128

model.fit(train_img, train_labels, epochs=epochs, batch_size=batchsize)
model.evaluate(test_img, test_labels, batch_size=batchsize)
model.save(f"{BASE_DIR}/CNN_2")

# epochs=20; batchsize=128; learning_rate=0.0001; layers.Dense=64; loss: 3.1286->3.2206 	acc: 0.2527->0.2441
# epochs=40; batchsize=256; learning_rate=0.0001; layers.Dense=64; loss: 2.9925->3.1148 	acc: 0.2793->0.2576
# epochs=30; batchsize=128; learning_rate=0.0001; layers.Dense=64; loss: 2.9516->3.0678 	acc: 0.2844->0.2693
# epochs=40; batchsize=128; learning_rate=0.0001; layers.Dense=64; loss: 2.8247->2.9871 	acc: 0.3082->0.2829
# epochs=20; batchsize=128; learning_rate=0.0005; layers.Dense=32; loss: 2.6466->2.7938 	acc: 0.3345->0.3052
# epochs=30; batchsize=128; learning_rate=0.0005; layers.Dense=32; loss: 2.6195->2.7846 	acc: 0.3353->0.3081
# epochs=20; batchsize=256; learning_rate=0.0005; layers.Dense=64; loss: 2.5952->2.8108 	acc: 0.3515->0.3122
# epochs=40; batchsize=256; learning_rate=0.0005; layers.Dense=32; loss: 2.5107->2.7026 	acc: 0.3640->0.3288
# epochs=30; batchsize=128; learning_rate=0.0005; layers.Dense=64; loss: 2.2981->2.6127 	acc: 0.4112->0.3429
# epochs=20; batchsize= 64; learning_rate=0.0005; layers.Dense=64; loss: 2.2824->2.5878 	acc: 0.4143->0.3508
# epochs=20; batchsize=128; learning_rate=0.0005; layers.Dense=64; loss: 2.3486->2.6018 	acc: 0.4033->0.3515
# epochs=20; batchsize= 64; learning_rate= 0.001; layers.Dense=64; loss: 2.0670->2.6472 	acc: 0.4593->0.3566
# epochs=20; batchsize=128; learning_rate=0.0005; layers.Dense=128+64; loss: 2.1978->2.5731 acc: 0.4277->0.3593
# epochs=40; batchsize=128; learning_rate= 0.001; layers.Dense=64; loss: 2.0049->2.5931 	acc: 0.4697->0.3617
# epochs=40; batchsize= 64; learning_rate=0.0005; layers.Dense=64; loss: 2.0725->2.5664 	acc: 0.4577->0.3658
# epochs=40; batchsize= 64; learning_rate= 0.001; layers.Dense=64; loss: 1.8451->2.6522 	acc: 0.5058->0.3659
# epochs=40; batchsize=256; learning_rate=0.0005; layers.Dense=64; loss: 2.1814->2.5574 	acc: 0.4401->0.3666
# epochs=40; batchsize=128; learning_rate=0.0005; layers.Dense=64; loss: 1.9997->2.5098 	acc: 0.4774->0.3770

