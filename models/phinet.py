from keras.layers import Input, Conv2D, Reshape, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import add, concatenate

def phinet(input_shape, n_classes, learning_rate=1e-3, weights=None):
    # be sure to add channel to the binary tensor
    inputs = Input(input_shape)
    reshape = Reshape(input_shape + (1,))
    init = Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', )(reshape)
    init = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(init)

    # linear segment
    x = Conv2D(32, (3,3), strides=(1,1), padding='same')(init)
    x = Conv2D(32, (3,3), strides=(1,1), padding='same')(x)
    x = Conv2D(32, (3,3), strides=(1,1), padding='same')(x)

    # residual segment
    y = Conv2D(32, (3,3), strides=(1,1), padding='same')(init)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    z = Conv2D(32, (3,3), strides=(1,1), padding='same')(y)
    y = add([y,z])
    y = Activation('relu')(y)

    for _ in range(5):
        y = Conv2D(32, (3,3), strides=(1,1), padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        z = Conv2D(32, (3,3), strides=(1,1), padding='same')(y)
        y = add([y,z])
        y = Activation('relu')(y)

    # pooling segment
    z = AveragePooling2D(pool_size=(7,7), strides=(1,1), padding='same')(init)
    z = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(z)

    # concatenate
    x = concatenate([x,y,z], axis=3)
    x = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(n_classes, activation='softmax')(x)


    model = Model(inputs = inputs, outputs = out)
    if weights:
        model.load_weights(weights)

    model.compile(optimizer=Adam(lr=learning_rate),\
            loss='categorical_crossentropy', metrics=['accuracy'])
    return model
