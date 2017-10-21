from keras.models import Model
from keras.layers import Conv2D, Input, Reshape, Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import add


# Input Shape will be 1024 x 1024
def convNeuralNet(input_shape, num_classes, lrate = 1e-3, weights = None):
    input_img = Input(shape=(input_shape))

    # Reshape Layer
    reshape = Reshape((input_shape, 1))(input_img)

    conv1 = Conv2D(32, (3,3), strides=(1,1), padding='same')(reshape)

    # residual layer
    conv2 = Conv2D(32, (3,3), strides=(1,1), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(32, (3,3), strides=(1,1), padding='same')(conv2)
    conv2 = add([conv2, conv3])
    conv2 = Activation('relu')(conv2)

    for _ in range (5):
        conv2 = Conv2D(32, (3,3), strides=(1,1), padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv3 = Conv2D(32, (3,3), strides=(1,1), padding='same')(conv2)
        conv2 = add([conv2, conv3])
        conv2 = Activation('relu')(conv2)

    # pooling layer

    conv2 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(conv2)

    # Flatten layer
    conv2 = Flatten()(conv2)

    # Dense output layer
    out = Dense(num_classes, activation='softmax')(conv2)

    model = Model(inputs= reshape, outputs= out)
    if weights:
        model.load_weights(weights)

    model.compile(optimizer=(Adam(lr=lrate)), loss="categorical_crossentropy", metrics=['accuracy'])

    return model


