import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM, ConvLSTM2D, Conv2D, MaxPooling3D, Dropout, Flatten, Conv3D,Convolution3D, ZeroPadding3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
# constants and parameters
LAYER_HEIGHT = 14
LAYER_WIDTH = 240
LAYER_LENGTH = 160
NUM_CLASS = 119

def cnn_model():
    adam = Adam()
    input_shape = (LAYER_HEIGHT, LAYER_WIDTH, LAYER_LENGTH, 1)
    model = Sequential()


    model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=(LAYER_HEIGHT, LAYER_WIDTH, LAYER_LENGTH, 1)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='same', name='pool5'))
	
	#fully connected
    model.add(Flatten())
	#fully connected => 4096
    model.add(Dense(4096, activation='relu'))
	#fully connected => 2048
    model.add(Dense(2048, activation='relu'))
	#fully connected => 1024
    model.add(Dense(1024, activation='relu'))
	#softmax => 119
    # model.add(Dense(NUM_CLASS, activation='softmax'))
	# model = multi_gpu_model(model, gpus=2)
    # model.compile(loss=[keras.losses.categorical_crossentropy],
    #           optimizer=adam,
    #           metrics=['accuracy'])


    return model, adam
