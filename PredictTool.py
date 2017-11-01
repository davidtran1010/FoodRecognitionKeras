import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import array
model = load_model('/Users/davidtran/Downloads/trainedModel13.h5')


# test_data_dir = '/Users/davidtran/Downloads/Google_Images/res_128x128 copy/validation'
# train_datagen = ImageDataGenerator()
#
# test_datagen = ImageDataGenerator()
#
# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(128, 128),
#     batch_size=16,
#     class_mode='categorical')
#
# print test_generator.class_indices
#
# metrics = model.evaluate_generator(test_generator,50,workers=20)
# print('')
# #print(np.ravel(model.predict(train_tensors)))
# print('training data results: ')
# for i in range(len(model.metrics_names)):
#     print(str(model.metrics_names[i]) + ": " + str(metrics[i]))


#img = Image.open("/Users/davidtran/Downloads/test.jpg")
img = np.array(Image.open("/Users/davidtran/Downloads/banhmi 27.jpg"),dtype=np.uint8)
img = img.reshape(1,150,150,3)
score = model.predict(img, batch_size=32, verbose=0)
classes = np.argmax(score,axis=1)

print "Note: {'Pho': 8, 'ComTam': 5, 'HuTieu': 6, 'BanhCanh': 0, 'BunRieu': 4, 'MiQuang': 7, 'BanhMi': 1, 'BoNe': 2, 'BunBo': 3}"
print "{'Tao': 12, 'Pho': 11, 'ComTam': 6, 'Nho': 10, 'Chuoi': 5, 'HuTieu': 8, 'BanhCanh': 0, 'BunRieu': 4, 'DuaHau': 7, 'MiQuang': 9, 'BanhMi': 1, 'BoNe': 2, 'BunBo': 3}"

print classes

#img = cv2.imread('/Users/davidtran/Downloads/Google_Images/res_128x128 copy/train/BunBo/BunBo_5.jpg')
#img = np.array([img]).reshape((3,128, 128))
#print(model.summary()) # model is already trained