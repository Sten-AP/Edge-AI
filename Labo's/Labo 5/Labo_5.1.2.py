import keras
import os
import numpy
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

BASE_DIR = os.path.dirname(__file__)

cifar10 = keras.datasets.cifar10
(train_img, train_labels), (test_img, test_labels) = cifar10.load_data( )
ctg_names = [ "vliegtuig", "auto", "vogel", "kat", "hert", "hond", "kikker", "paard", "schip", "vrachtwagen" ]

new_model = keras.models.load_model(f"{BASE_DIR}/CNN_1")

index = 2

plt.imshow(test_img[index], cmap=plt.cm.binary)
plt.xlabel(ctg_names[test_labels[index][0]] )
plt.show()

prediction = new_model.predict(test_img)
predic_img = (numpy.argmax(prediction[index]))
print(ctg_names[predic_img])