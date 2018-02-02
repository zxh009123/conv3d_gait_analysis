from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import cnn_model as network
from cnn_model import cnn_model
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM, ConvLSTM2D, Conv2D, MaxPooling3D, Dropout, Flatten, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt

# constants and parameters
BATCH_SIZE = 1
NUM_CLASS = network.NUM_CLASS
MAX_LR = 0.001
MIN_LR = 0.0001
EPOCHS = 10
LR_RATIO = (MAX_LR - MIN_LR) / EPOCHS
LAYER_HEIGHT = network.LAYER_HEIGHT
LAYER_WIDTH = network.LAYER_WIDTH
LAYER_LENGTH = network.LAYER_LENGTH
NUM_TRAIN = 59826 - NUM_CLASS * LAYER_HEIGHT
NUM_TEST = 11905 - NUM_CLASS * LAYER_HEIGHT

#taining, testing samples and lables
x_test = np.ndarray(shape=(int(NUM_TEST), LAYER_HEIGHT, LAYER_WIDTH, LAYER_LENGTH, 1), dtype=np.float32)

y_test_lable = np.zeros(int(NUM_TEST),dtype=int)

#weight path, sample path
SAMPLE_PATH = '/home/xiaohan/Downloads/CASIA/DatasetB/silhouettes/'
CKPT_PATH = './model/weights.best.hdf5'
FOLDER_NAME = ['/nm-01/000/', '/nm-02/000/', '/nm-03/000/', '/nm-04/000/', '/nm-05/000/', '/nm-06/000/']

folder = [f for f in listdir(SAMPLE_PATH)]
folder.sort()
print('total objects : '+str(len(folder)))

cat_num = 0
test_index = 0

#build the test set
for obj in folder:
	files = [img for img in listdir(SAMPLE_PATH + obj + FOLDER_NAME[5])]
	files.sort()
	for i in range(0, int(len(files) - LAYER_HEIGHT)):
		images_test = np.ndarray(shape=( LAYER_HEIGHT, LAYER_WIDTH, LAYER_LENGTH, 1), dtype=float)
		for image_num in range(0, LAYER_HEIGHT):
			image = cv2.imread(SAMPLE_PATH + obj + FOLDER_NAME[5] + files[i], 0)
			# print(SAMPLE_PATH + obj + FOLDER_NAME[5] + files[i])
			image = image.reshape((LAYER_WIDTH,LAYER_LENGTH,1))
			images_test[image_num] = image
		y_test_lable[test_index] = cat_num
		x_test[test_index] = images_test
		test_index += 1
	cat_num += 1

x_test /= 255

max_step = 119 * 5 * (EPOCHS - 2)
current_step = 0
acc = 0

model = cnn_model()
y_test_lable = keras.utils.to_categorical(y_test_lable, NUM_CLASS)

val_acc = []
val_loss = []
epoch_number = []

# collecting the training data and training
print('-------------start training---------------')
for e in range(EPOCHS):
	epoch_number.append(e)
	cat_num = 0
	print('---------------------epoch : '+str(e))
	for obj in folder:
		for file_name in FOLDER_NAME[0:4]:
			files = [img for img in listdir(SAMPLE_PATH + obj + file_name)]
			files.sort()
			y_train_lable = np.zeros(int(len(files) - LAYER_HEIGHT),dtype=int)
			x_train = np.ndarray(shape=(int(len(files) - LAYER_HEIGHT), LAYER_HEIGHT, LAYER_WIDTH, LAYER_LENGTH, 1), dtype=np.float32)
			train_index = 0
			for i in range(0, int(len(files) - LAYER_HEIGHT)):
				images_train = np.ndarray(shape=( LAYER_HEIGHT, LAYER_WIDTH, LAYER_LENGTH, 1), dtype=float)
				for image_num in range(0, LAYER_HEIGHT):
					image = cv2.imread(SAMPLE_PATH + obj + file_name + files[i], 0)
					image = image.reshape((LAYER_WIDTH,LAYER_LENGTH,1))
					images_train[image_num] = image
				y_train_lable[train_index] = cat_num
				x_train[train_index] = images_train
				train_index += 1
			x_train /= 255
			y_train_lable = keras.utils.to_categorical(y_train_lable, NUM_CLASS)
			history = model.fit(x_train, y_train_lable,
          				batch_size=BATCH_SIZE,
          				epochs=1,
          				verbose=0)
			current_step += 1
			print('progress : '+str(current_step/max_step * 100))
		cat_num += 1
	score = model.evaluate(x_test, y_test_lable, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	val_acc.append(score[1])
	val_loss.append(score[0])
	if score[1] > acc:
		model.save_weights("119-obj-best.hdf5")
		acc = score[1]
		print('save best model')

#show plot of loss and accuracy
plt.figure(212)
plt.subplot(212) 
plt.plot(epoch_number,val_loss,label='loss')
plt.legend(loc='upperleft')
plt.subplot(211) 
plt.plot(epoch_number,val_acc,lable='accuracy')
plt.legend(loc='upperleft')
plt.show()
