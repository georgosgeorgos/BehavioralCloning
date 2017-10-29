from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Flatten, Dense, Activation, Dropout, Lambda, Cropping2D, Convolution2D


def Net(input_shape, crop):

    model = Sequential()
    model.add(Cropping2D(cropping=crop, input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5)) 
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation="relu", init="he_normal"))
    model.add(BatchNormalization())
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation="relu", init="he_normal"))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation="relu", init="he_normal"))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2,2), activation="relu", init="he_normal"))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2,2), activation="relu", init="he_normal"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu', init="he_normal"))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu', init="he_normal"))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='relu', init="he_normal"))
    model.add(BatchNormalization())
    model.add(Dense(1))
    
    return model