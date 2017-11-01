from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import datetime
import time
import coremltools
from sklearn.model_selection import train_test_split

import numpy as np
import cPickle
import h5py
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt





# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/Users/davidtran/Downloads/Google_Images/res_150x150/train'
validation_data_dir = '/Users/davidtran/Downloads/Google_Images/res_150x150/validation'
nb_train_samples = 900
nb_validation_samples = 100
epochs = 1
batch_size = 25



# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(150,150,3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(9, activation='softmax',name='predictions')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')



print "[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
start = time.time()
# train the model on the new data for a few epochs
# model.fit_generator(
#     train_generator,
#     samples_per_epoch=nb_train_samples,
#     epochs=1,
#     validation_data=validation_generator,
#     nb_val_samples=nb_validation_samples)
# end time
end = time.time()
print "[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
print "[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
start = time.time()

model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
# end time
end = time.time()
print "[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


# metrics = model.evaluate_generator(train_generator,)
# print('')
# #print(np.ravel(model.predict(train_tensors)))
# print('training data results: ')
# for i in range(len(model.metrics_names)):
#     print(str(model.metrics_names[i]) + ": " + str(metrics[i]))


print "Creating model..."
#save to mlmodel
labels = ["Pho","ComTam","HuTieu","BanhCanh","BunRieu","MiQuang","BanhMi","BoNe","BunBo"]



model.save("/Users/davidtran/PycharmProjects/tensorflowtest/flower_keras/output/trainedModel.h5")