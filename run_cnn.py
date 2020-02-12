import json
from sklearn.metrics import classification_report
from keras.callbacks import Callback
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
import os
import math
import time
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import cv2
from keras import backend as K


def normalize(x):
    return (x+1e-10) / (np.sqrt(np.mean(np.square(x)))+1e-10)

def load_images(images_json):
    train_files = []
    y_train = []
    for image_path in images_json:
        train_files.append(image_path)
        label = images_json[image_path]
        y_train.append(int(label))

    width = 256
    height = 256
    dataset = np.ndarray(shape=(len(train_files),width,height,3),dtype=np.float32)
    
    i = 0
    start = time.time()
    for _file in train_files:
        if os.path.exists(f"scan_jpgs/{_file}"):
            img = load_img(f"scan_jpgs/{_file}")
            #img.thumbnail((width,height))
            x_copy = img_to_array(img).copy()
            x = cv2.resize(x_copy,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
            x = normalize(x)
        else:
            print(f"{_file} doesn't exist.")
        

        dataset[i] = x
        i += 1

    # Save to a file
    y_train = np.array(y_train)
    #np.save("train_data.npy",dataset)
    #np.save("y_train.npy",y_train)


    X_train, X_test, y_train, y_test = train_test_split(dataset, y_train, test_size=0.2, random_state=33)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=33)
    print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(X_train), len(X_val), len(X_test)))
    return X_train,y_train,X_val,y_val,X_test,y_test

def create_data():
    dataset = np.load('databases/train_data.npy')
    y_train = np.load('databases/y_train.npy')

    X_train, X_test, y_train, y_test = train_test_split(dataset, y_train, test_size=0.2, random_state=33)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=33)
    print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(X_train), len(X_val), len(X_test)))
    return X_train,y_train,X_val,y_val,X_test,y_test
"""
with open('databases/image_jpg_labelled.json') as f:
    blah = json.loads(f.read())
    load_images(blah)
"""

if __name__ == "__main__":
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='tanh',
                 input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])

    with open('databases/image_jpg_labelled.json') as f:
        blah = json.loads(f.read())
        x_train,y_train,x_val,y_val,x_test,y_test = load_images(blah)

    print("STARTING TRAING MATHAFUCKA!")
    batch_size = 128
    num_classes = 10
    epochs = 3

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('model.h5')
    y_pred = model.predict(x_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_bool))
