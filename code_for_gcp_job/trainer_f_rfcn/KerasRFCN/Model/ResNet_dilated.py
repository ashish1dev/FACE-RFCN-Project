"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

'''
This is Backbone of RFCN Model
Dilated ResNet50 or 101
Paper: DetNet: A Backbone network for Object Detection
https://arxiv.org/abs/1804.06215
'''

import keras.layers as KL

class ResNet_dilated(object):
    """docstring for ResNet101"""
    def __init__(self, input_tensor, architecture='resnet50'):
        self.keras_model = ""
        self.input_tensor = input_tensor
        self.output_layers = ""
        assert architecture in ['resnet50', 'resnet101'], 'architecture must be resnet50 or resnet101!'
        self.architecture = architecture
        self.construct_graph(input_tensor)
        
    def construct_graph(self, input_tensor, stage5=True):
        assert self.input_tensor is not None, "input_tensor can not be none!"
        # Stage 1
        x = KL.ZeroPadding2D((3, 3))(input_tensor)
        x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
        x = BatchNorm(axis=3, name='bn_conv1')(x)
        x = KL.Activation('relu')(x)
        C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        # Stage 2
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        C2 = x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        # Stage 3
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        C3 = x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        # Stage 4
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        block_count = {"resnet50": 5, "resnet101": 22}[self.architecture]
        for i in range(block_count):
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
        C4 = x
        # Stage 5
        x = self.conv_block(x, 3, [256, 256, 256], stage=5, block='a', dilated=2, strides=(1, 1))
        x = self.identity_block(x, 3, [256, 256, 256], stage=5, block='b', dilated=2)
        C5 = x = self.identity_block(x, 3, [256, 256, 256], stage=5, block='c', dilated=2)
        # Stage 6
        x = self.conv_block(x, 3, [256, 256, 256], stage=6, block='a', dilated=2, strides=(1, 1))
        x = self.identity_block(x, 3, [256, 256, 256], stage=6, block='b', dilated=2)
        C6 = x = self.identity_block(x, 3, [256, 256, 256], stage=6, block='c', dilated=2)

        P6 = KL.Conv2D(256, (1, 1), name='fpn_c6p6')(C6)
        P5 = KL.Add(name="fpn_p5add")([P6, KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)])
        P4 = KL.Add(name="fpn_p4add")([P5, KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])

        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p6")(P6)

        self.output_layers = [P2, P3, P4, P5, P6]

    def conv_block(self, input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, dilated=1):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias, dilation_rate=dilated)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                      '2c', use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

        shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
        shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    def identity_block(self, input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, dilated=1):
        """The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      use_bias=use_bias)(input_tensor)
        x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias, dilation_rate=dilated)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)