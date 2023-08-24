from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2
import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2


class InceptionResNetV2UNet:
    def __init__(self, input_size=256):
        self.input_size = input_size

    def build_model(self):
        def conv_block(input, num_filters):
            x = Conv2D(num_filters, 3, padding="same")(input)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(num_filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            return x

        def decoder_block(input, skip_features, num_filters):
            x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
            x = Concatenate()([x, skip_features])
            x = conv_block(x, num_filters)
            return x

        """ Input """
        inputs = Input((self.input_size, self.input_size, 3))

        """ Pre-trained InceptionResNetV2 Model """
        encoder = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)

        """ Encoder """
        s1 = encoder.layers[0].output  ## (512 x 512)

        s2 = encoder.layers[3].output  ## (255 x 255)
        s2 = ZeroPadding2D(((1, 0), (1, 0)))(s2)  ## (256 x 256)

        s3 = encoder.layers[13].output  ## (126 x 126)
        s3 = ZeroPadding2D((1, 1))(s3)  ## (128 x 128)

        s4 = encoder.layers[265].output  ## (61 x 61)
        s4 = ZeroPadding2D(((2, 1), (2, 1)))(s4)  ## (64 x 64)

        """ Bridge """
        b1 = encoder.layers[501].output  ## (30 x 30)
        b1 = ZeroPadding2D((1, 1))(b1)  ## (32 x 32)

        """ Decoder """
        d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
        d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
        d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
        d4 = decoder_block(d3, s1, 64)  ## (512 x 512)

        """ Output """
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
        print(outputs)
        model = Model(inputs, outputs, name="InceptionResNetV2_UNet")
        return model
# if __name__ == "__main__":
#     # model = InceptionResNetV2UNet()
#     # print(input_size)
#     from torchsummary import summary
#     summary(model, (256,256,3))
