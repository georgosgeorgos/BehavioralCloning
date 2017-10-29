from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Merge, merge, Flatten, Dense, Activation, Dropout, Lambda, Cropping2D, Convolution2D, MaxPooling2D, GlobalAveragePooling2D
BatchNormalization()


def fire_module(x, s1, e1, e3):
    
    x = Convolution2D(s1, 1, 1, border_mode='valid', init="he_normal")(x)
    y = Convolution2D(e1, 1, 1, border_mode='valid', init="he_normal")(x)
    y = Activation('relu')(y)
    z = Convolution2D(e3, 1, 1, border_mode='same', init="he_normal")(x)
    z = Activation('relu')(z)
    x = merge([y, z], mode='concat', concat_axis=3)
    return x


def SqueezeNet(input_shape, crop):
    
    '''squeezeNet model with bypass'''
    
    In = Input(shape=input_shape)
    x = Cropping2D(cropping=crop, input_shape=input_shape)(In)
    x = Lambda(lambda x: (x / 255.0) - 0.5)(x)
    
    x = Convolution2D(48, 1, 1, border_mode='valid', activation="relu", init="he_normal")(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(x)
    x = BatchNormalization()(x)
    
    f2 = fire_module(x, 16, 32, 32)
    f3 = fire_module(f2, 16, 32, 32)
    x = merge([f2, f3], mode='concat', concat_axis=0)
    x = BatchNormalization()(x)
    f4 = fire_module(x, 16, 64, 64)
    
    f4 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(f4)
    f5 = fire_module(f4, 16, 64, 64)
    x = merge([f4, f5], mode='concat', concat_axis=0)
    x = BatchNormalization()(x)

    f6 = fire_module(x, 16, 128, 64)
    f7 = fire_module(f6, 16, 128, 64)
    x = merge([f6, f7], mode='concat', concat_axis=0)
    x = BatchNormalization()(x)
    f8 = fire_module(x, 16, 128, 128)
    
    f8 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(f8)
    f9 = fire_module(f8, 16, 128, 128)
    x = merge([f8, f9], mode='concat', concat_axis=0)
    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)
    x = Convolution2D(100, 1, 1, border_mode='valid', activation="relu", init="he_normal")(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    Out = Activation('linear')(x)
    
    squeeze = Model(input=In, output=Out)
    return squeeze