# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:14:59 2018

@author: ssurya200
"""
# Convolution Neural Network

# importing the packages for CNN from keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# initialising

def build_classifer(optimizer):
    classifier = Sequential()
    
    #first layer convolution with relu activation
     
    classifier.add(layer = Conv2D(32, (5, 5), padding='same', input_shape = (64, 64, 3), activation = "relu"))
    
    # second layer with pooling step - down sizing 
    
    classifier.add(layer = MaxPooling2D(pool_size = (2,2)))
 
    
    # repeat first 2 layers to increase the accuracy in test set
    
    classifier.add(layer = Conv2D(64, (5, 5), padding='same', activation = "relu"))
    
    classifier.add(layer = MaxPooling2D(pool_size = (2,2)))

    
    # adding one more conv layer with higher units
    '''
    classifier.add(layer = Conv2D(64, (3, 3), padding='same', activation = "relu"))
    
    classifier.add(layer = MaxPooling2D(pool_size = (2,2)))
    
    classifier.add(layer = Conv2D(64, (3, 3), padding='same', activation = "relu"))
    
    classifier.add(layer = MaxPooling2D(pool_size = (2,2)))'''
    
    # thrid layer flattening
    
    classifier.add(layer = Flatten())
    
    # fourht layer ANN - fully connected layer
    
    classifier.add(Dense(units = 1024, 
                         activation = "relu"))
    classifier.add(Dropout(rate = 0.4))
    '''
    classifier.add(Dense(units = 64, 
                         activation = "relu"))
    classifier.add(Dropout(rate = 0.2))
    
    classifier.add(Dense(units = 32, 
                         activation = "relu"))
    classifier.add(Dropout(rate = 0.1))'''
    # fifth layer ANN - out put later
    
    classifier.add(Dense(units = 1, activation = "sigmoid"))
    
    
    # complie the model
    
    classifier.compile(optimizer = "adam",
                       loss = "binary_crossentropy",
                       metrics = ["accuracy"])

    return classifier
# fitting the model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

'''
classifier = KerasClassifier(build_fn = build_classifer)

parameters = {'batch_size' : [25, 32],
              'epochs' : [25, 50],
              'optimizer' : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
'''



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000/32))
    '''                     max_queue_size=10,
                         workers=3,
                         use_multiprocessing = True)



y_pred = classifier.predict(test_set)

classifier.predict()'''


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

result.shape()