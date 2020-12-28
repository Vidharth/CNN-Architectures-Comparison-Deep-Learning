from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.activations import softmax, softplus, sigmoid, relu, elu
from tensorflow.keras.layers import LeakyReLU, PReLU
import numpy as np
import os

def run():
	img_width, img_height = 150, 150

	base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
	model = Sequential()
	model.add(base_model)
	model.add(Flatten())
	model.add(Dense(4096))
	model.add(BatchNormalization())
	model.add(Activation('softplus'))
	model.add(Dropout(0.2))
	model.add(Dense(4096))
	model.add(BatchNormalization())
	model.add(Activation('softplus'))
	model.add(Dropout(0.2))
	model.add(Dense(6, activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	image_folder = '/home/s3911888/image-classification'
	batchSize = 256
	datagen = image.ImageDataGenerator()

	train_it = datagen.flow_from_directory(image_folder+'/seg_train/seg_train/',
										   target_size=(img_width, img_height),
										   color_mode='rgb', class_mode='categorical',
										   batch_size=batchSize)
	test_it = datagen.flow_from_directory(image_folder+'/seg_test/seg_test/',
										   target_size=(img_width, img_height),
										   color_mode='rgb', class_mode='categorical',
										   batch_size=batchSize)
	model.fit(train_it, epochs=10, verbose=1)
	training_loss = model.evaluate_generator(train_it)
	print("Training loss", training_loss)
	model.predict_generator(test_it)
	test_loss = model.evaluate_generator(test_it)
	print("Testing loss", test_loss)
	return training_loss, test_loss

if __name__ == '__main__':
		training_loss, testing_loss = run()
		with open('vgg_output', 'a') as f:
			f.writelines(str(training_loss)+'\n')
			f.writelines(str(testing_loss)+'\n')
