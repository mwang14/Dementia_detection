from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys
import cv2
import numpy as np

model = load_model('model.h5')


model.summary()
for ilayer, layer in enumerate(model.layers):
    print("{:3.0f} {:10}".format(ilayer, layer.name))

img = load_img(sys.argv[1])
x = img_to_array(img).copy()

res = cv2.resize(x, dsize=(256,256), interpolation=cv2.INTER_CUBIC)

