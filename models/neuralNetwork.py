from keras.models import Model
from keras.layer import Conv2D, Input, Reshape, MaxPooling2D, Activation


# Input Shape will be 1024 x 1024
def convNeuralNet(input_shape, lrate = 1e-3):
    input_img = Input(shape=(None, input_shape, input_shape, 1))
    conv1 = Conv2D(32, [3,3], strides=(1,1), padding='same')(input_img)

    # residual segment
    conv2 = Conv2D(32, [3,3], strides=(1,1), padding='same')(conv1)
    conv2 = BatchNormalization(conv2)
    conv2 = activation('relu')(conv2)
    conv3 = Conv2D(32, [3,3], strides=(1,1), padding='same')(conv2)
    conv2 = add([conv2, conv3])
    conv2 = activation('relu')(conv2)

    for _ in range (5):
        conv2 = Conv2D(32, [3,3], strides=(1,1), padding='same')(conv2)
        conv2 = BatchNormalization(conv2)
        conv2 = activation('relu')(conv2)
        conv3 = Conv2D(32, [3,3], strides=(1,1), padding='same')(conv2)
        conv2 = add([conv2, conv3])
        conv2 = activation('relu')(conv2)

    # pooling segment



