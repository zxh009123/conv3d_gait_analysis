import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM, ConvLSTM2D, Conv2D, MaxPooling3D, Dropout, Flatten, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# constants and parameters
LAYER_HEIGHT = 5
LAYER_WIDTH = 240
LAYER_LENGTH = 300
NUM_CLASS = 119

def cnn_model():
	adam = Adam(lr = 1e-5)
	input_shape = (LAYER_HEIGHT, LAYER_WIDTH, LAYER_LENGTH, 1)
	model = Sequential()
	#conv1
	model.add(Conv3D(16, (2,2,2), strides=(1, 1, 1), padding='same',data_format = 'channels_last' , dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=(LAYER_HEIGHT, LAYER_WIDTH, LAYER_LENGTH, 1)))
	model.add(Dropout(0.25))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

	#conv2
	model.add(Conv3D(32, (2,2,2), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.add(Dropout(0.25))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	#conv3
	model.add(Conv3D(64, (2,2,2), strides=(1, 1, 1), padding='same', data_format = 'channels_last', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.add(Dropout(0.25))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	#conv4
	model.add(Conv3D(128, (2,2,2), strides=(1, 1, 1), padding='same', data_format = 'channels_last', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.add(Dropout(0.25))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	#fully connected
	model.add(Flatten())
	model.add(Dense(512, activation='tanh'))
	model.add(Dense(NUM_CLASS, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adam,
              metrics=['accuracy'])
	return model
