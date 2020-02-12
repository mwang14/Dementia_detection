import SimpleITK as sitk
import json
import numpy
#import tensorflow as tf
import sys
import logging
import os
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib

def read_image(img_path):
    sitk_t1 = sitk.ReadImage(img_path)
    t1 = sitk.GetArrayFromImage(sitk_t1)
    return t1

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tf_records(images_json):
    train_filename = 'train.tfrecords'
    writer = tf.io.TFRecordWriter(train_filename)
    for img_path in images_json:
        if os.path.exists(img_path): 
            img = read_image(img_path)   
            print(f"now reading {img_path}")
            label = images_json[img_path]
            feature = {'train/label': _int64_feature(label),
                'train/image': _float_feature(img.ravel())}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    #with open(sys.argv[1]) as f:
    #    images_json = json.load(f)
    #create_tf_records(images_json)
    a = nib.load('scans/OAS30101_MR_d0101/anat2/sub-OAS30101_ses-d0101_run-01_T1w.nii')
    #print(a[:,:,0].shape)
    img_data = a.get_data()
    img = Image.fromarray(img_data[:,:,0], 'L')
    img.save('image.jpeg')
