"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

'''
This is a demo to TRAIN a RFCN model with DeepFashion Dataset
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
'''

from KerasRFCN.Model.Model import RFCN_Model
from KerasRFCN.Config import Config
from KerasRFCN.Utils import Dataset
import os
import pickle
import numpy as np
from PIL import Image

import keras as keras
from keras.models import Model
from keras.layers import Input, Dense, Activation,Flatten, Conv2D
from keras.layers import Dropout, MaxPooling2D
import keras.backend as K
from keras import callbacks
import tensorflow as tf
K.set_image_data_format('channels_last')
import numpy as np
import argparse
from tensorflow.python.lib.io import file_io

############################################################
#  Config
############################################################

class RFCNNConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Fashion"

    # Backbone model
    # choose one from ['resnet50', 'resnet101', 'resnet50_dilated', 'resnet101_dilated']
    BACKBONE = "resnet101"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    C = 1 + 1  # background + 2 tags
    NUM_CLASSES = C
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 768

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    # Use same strides on stage 4-6 if use dilated resnet of DetNet
    # Like BACKBONE_STRIDES = [4, 8, 16, 16, 16]
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200

    RPN_NMS_THRESHOLD = 0.6
    POOL_SIZE = 7






def main(job_dir,**args):

    ##Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard'

    ##Using the GPU
    with tf.device('/device:GPU:0'):
        ROOT_DIR = os.getcwd()

        file = job_dir + "WIDERFACE_DATA/wider_face_split/" + "wider_face_train_bbx_gt.txt";
        file1 =  job_dir + "WIDERFACE_DATA/WIDER_train/images/";
        dataset_train =  Dataset(file, file1)
        config = RFCNNConfig()


        # Validation dataset
        file2 =  job_dir + "WIDERFACE_DATA/wider_face_split/" + "wider_face_val_bbx_gt.txt"
        file3 =  job_dir + "WIDERFACE_DATA/WIDER_val/images/";
        dataset_val = Dataset(file, file1)

        #
        # model = RFCN_Model(mode="training", config=config, model_dir=os.path.join(ROOT_DIR, "logs") )
        #
        # # This is a hack, bacause the pre-train weights are not fit with dilated ResNet
        # model.keras_model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True, skip_mismatch=True)
        #
        # try:
        #     model_path = model.find_last()[1]
        #     if model_path is not None:
        #         model.load_weights(model_path, by_name=True)
        # except Exception as e:
        #     print(e)
        #     print("No checkpoint founded")
        #
        # # *** This training schedule is an example. Update to your needs ***
        #
        # # Training - Stage 1
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=20,
        #             layers='heads')
        #
        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=40,
        #             layers='4+')
        #
        # # Training - Stage 3
        # # Fine tune all layers
        # print("Fine tune all layers")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=80,
        #             layers='all')
        #
        # # Training - Stage 3
        # # Fine tune all layers
        # print("Fine tune all layers")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=240,
        #             layers='all')



##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
