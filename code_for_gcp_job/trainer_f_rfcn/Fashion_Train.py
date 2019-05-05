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

############################################################
#  Dataset
############################################################

if __name__ == '__main__':
    ROOT_DIR = "code_for_gcp_job/trainer_f_rfcn/" #'gs://bucket1cse/' #os.getcwd()
    # file = r"E:\2nd Assignment\Final Project\wider_face_split\\" + "wider_face_train_bbx_gt.txt"
    # file1 = r"E:\2nd Assignment\Final Project\WIDER_train\images\\"

    # img_path = '../WIDERFACE_DATA/WIDER_train/images/0--Parade/0_Parade_Parade_0_3.jpg'

    job_dir = "code_for_gcp_job/trainer_f_rfcn/" # "gs://bucket1cse/"
    file = job_dir + "WIDERFACE_DATA/wider_face_split/" + "wider_face_train_bbx_gt.txt"
    file1 = job_dir +  "WIDERFACE_DATA/WIDER_train/images/"

    print("file1 = ")
    print(file1)

    dataset_train =  Dataset(file, file1)
    config = RFCNNConfig()

    print("dataset_train = ")
    print((dataset_train.dfb.shape))
    # print(dataset_train.dfb[0])

    # Validation dataset
    # file2 =  r"E:\2nd Assignment\Final Project\wider_face_split\\"  + "wider_face_val_bbx_gt.txt"
    # file3 = r"E:\2nd Assignment\Final Project\WIDER_val\images\\"
    file2 = job_dir +  "WIDERFACE_DATA/wider_face_split/" + "wider_face_val_bbx_gt.txt"
    file3 = job_dir +  "WIDERFACE_DATA/WIDER_val/images/";
    dataset_val = Dataset(file2, file3)


    model = RFCN_Model(mode="training", config=config, model_dir=os.path.join(ROOT_DIR, "logs") )

    dir = "/home/ashish/workspace/FACE-RFCN-Project/code_for_gcp_job/trainer_f_rfcn/"
    # dir = "/Users/ashish/workspace/ub_coursework/second_semester/Biometrics_and_Image_Analysis/Project/FACE-RFCN-Project/code_for_gcp_job/trainer_f_rfcn/"

    # This is a hack, bacause the pre-train weights are not fit with dilated ResNet
    model.keras_model.load_weights(dir + "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True, skip_mismatch=True)

    try:
        model_path = model.find_last()[1]
        if model_path is not None:
            model.load_weights(model_path, by_name=True)
    except Exception as e:
        print(e)
        print("No checkpoint founded")

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=80,
                layers='all')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=240,
                layers='all')
