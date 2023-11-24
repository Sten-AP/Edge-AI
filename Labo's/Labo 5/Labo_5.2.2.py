import keras
import os
import numpy
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

BASE_DIR = os.path.dirname(__file__)

cifar100 = keras.datasets.cifar100
(train_img, train_labels), (test_img, test_labels) = cifar100.load_data( )
ctg_names = [  
'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard',
'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 
'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm' 
]
new_model = keras.models.load_model(f"{BASE_DIR}/CNN_2")
new_model.evaluate(test_img, test_labels, batch_size=512)

index = 0

plt.imshow(test_img[index], cmap=plt.cm.binary)
plt.xlabel(ctg_names[test_labels[index][0]] )
plt.show()

prediction = new_model.predict(test_img)
predic_img = (numpy.argmax(prediction[index]))
print(ctg_names[predic_img])