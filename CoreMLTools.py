import coremltools
from keras.models import load_model
#model = load_model('/Users/davidtran/Downloads/trainedModel.h5')
model = load_model('/Users/davidtran/Downloads/trainedModel.h5')
print model.summary()
import coremltools
import h5py




coreml_model = coremltools.converters.keras.convert(model,
                                                    input_names=['image'],
                                                    output_names=['probabilities'],
                                                    image_input_names='image',
                                                    class_labels='/Users/davidtran/PycharmProjects/tensorflowtest/flower_keras/classes.txt',
                                                    predicted_feature_name='class'
                                                    )

#coreml_model = coremltools.converters.keras.convert\
 #   ('/Users/davidtran/Downloads/trainedModel.h5',
  #   input_names='image',image_input_names = 'image',red_bias=-1.0, blue_bias=-1.0, green_bias=-1.0,
   #  image_scale=2 / 255.0,class_labels = output_labels,predicted_feature_name="predictions", is_bgr=True)
coreml_model.author = ''
coreml_model.short_description = ''
coreml_model.input_description['image'] = 'Vn Food'
print coreml_model
coreml_model.save('/Users/davidtran/Downloads/trainedModel.mlmodel')

