import glob
import numpy as np
import glob
import os
import shutil
import math
import pandas as pd
from datetime import datetime
from keras.layers import Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import keras
from keras import regularizers
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import  preprocess_input
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer,BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.activations import softmax

from sklearn.preprocessing import OneHotEncoder

from keras.utils import to_categorical

from keras.utils import plot_model
from keras.layers import Concatenate
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn import metrics


input_imgen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range=5.,
                                   horizontal_flip = True)

test_imgen = ImageDataGenerator(rescale = 1./255)



def generate_generator_multiple(generator,dir1, dir2, batch_size, img_height,img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=False,
                                          seed=7)

    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=False,
                                          seed=7)

    while True:
            X1i = genX1.next()
            '''print(X1i[0][0].shape)
            image1 = X1i[0][0]
            plt.imshow(image1)
            plt.show()'''
            X2i = genX2.next()
            '''image2 =  X2i[0][0]
            plt.imshow(image2)
            plt.show()'''
            yield [X1i[0], X2i[0]], X2i[1]


RGB_train_path='/home/sai/Desktop/dataset/RGB2/training/'
#RGB_test_path='/home/sai/Desktop/dataset/RGB/test/'
RGB_val_path='/home/sai/Desktop/dataset/RGB2/validation/'

depth_train_path='/home/sai/Desktop/dataset/depth2/training/'
#depth_test_path='/home/sai/Desktop/dataset/depth/test/'
depth_val_path='/home/sai/Desktop/dataset/depth2/validation/'





batch_size=64
img_height,img_width=(150,150)
inputgenerator=generate_generator_multiple(generator=input_imgen,
                                           dir1=RGB_train_path,
                                           dir2=depth_train_path,
                                           batch_size=batch_size,
                                           img_height=img_height,
                                           img_width=img_height)

valgenerator=generate_generator_multiple(test_imgen,
                                          dir1=RGB_val_path,
                                          dir2=depth_val_path,
                                          batch_size=batch_size,
                                          img_height=img_height,
                                          img_width=img_height)

'''test_generator=generate_generator_multiple(generator=test_imgen,
                                           dir1=RGB_test_path,
                                           dir2=depth_test_path,
                                           batch_size=1,
                                           img_height=150,
                                           img_width=150)
'''

print('generator_success')

'''RGB_test_datagen = ImageDataGenerator()
test_gen=RGB_test_datagen.flow_from_directory(RGB_test_path,
                                      target_size = (img_height,img_width),
                                      class_mode = 'categorical',
                                      batch_size = 1,
                                      shuffle=False,
                                      seed=7)
'''


inception_conv_C =  InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#Depth Model
inception_conv_D =  InceptionV3(weights='imagenet', include_top=False, input_shape= (150, 150, 3))
for layer in inception_conv_C.layers:
    layer.name = layer.name + str('_C')

for layer in inception_conv_D.layers:
    layer.name = layer.name + str('_D')

for layer in inception_conv_D.layers[:-4]:
    layer.trainable = False
for layer in inception_conv_C.layers[:-4]:
    layer.trainable = False
mergedModel = Concatenate()([inception_conv_C.output,inception_conv_D.output])
mergedModel = Flatten()(mergedModel)
mergedModel = Dense(units = 1024)(mergedModel)
mergedModel = BatchNormalization()(mergedModel)
mergedModel = Activation('relu')(mergedModel)
mergedModel = Dropout(0.2)(mergedModel)
mergedModel = Dense(units = 512)(mergedModel)
mergedModel = BatchNormalization()(mergedModel)
mergedModel = Activation('relu')(mergedModel)
mergedModel = Dropout(0.2)(mergedModel)
mergedModel = Dense(units = 3,activation = 'softmax')(mergedModel)

new_model = Model(
    [inception_conv_C.input, inception_conv_D.input], #model with two input tensors
    mergedModel                         #and one output tensor
)

new_model.summary()
plot_model(new_model, to_file='model.png')

new_model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history=new_model.fit_generator(inputgenerator,
                        epochs = 8,
                        steps_per_epoch=4,
                        validation_data = valgenerator,
                        validation_steps =4,
                        use_multiprocessing=True,
                        callbacks=[tensorboard_callback],
                        shuffle=False)

'''Y_pred = new_model.predict_generator(test_generator,steps=2000)
y_pred = np.argmax(Y_pred, axis=1)
counter=0
for i in range(len(y_pred)):
    print(y_pred[i])
    print('true label')
    #print(test_gen.classes[i])
    counter=counter+1



print(counter)'''
'''
RGB_test_datagen.flow_from_directory(RGB_test_path,batch_size=64)
cm = metrics.confusion_matrix(test_gen.classes, y_pred)

plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

new_model.save_weights('new_model.h5')
'''
