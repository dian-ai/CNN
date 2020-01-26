#Convolutional Neural Network
#They are great deeplearning models for computer vision, to classify images or videos

#now out dataset contains images, so we need to do image preprocessing
#in order to extract the dependent variable based on independent Vars: we need to
# name all the images(Dataset) into their categories + a number: Dog1, Cat1
# then write a code to extract the label name, to specify to the algorithm whether this image belongs to cat or dog

# a better solution is : to import images with Keras, we need to prepare a special structure for our dataset

# we already did the data preprocessing maually so only feature scaling is left which we do it later


#part 1 : building CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#images are 2d, videos are 3D
#to initialize we use sequential package

#initialize CNN
classifier = Sequential()

#building CNN is 4 steps, 1-Convolution,2-Maxpooling, 3-Flattening, 4-Full Connection

#Step 1 - Convolution: 
#input image (cat / dog) : Convolution applies several feature detectors on this input image, then we get a feature map, the highest numbers on feature map is where a feature detector could detect. number of feature detectors are the number of layers we are going to have in our convolution.(convolutional layer) video 55

classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation="relu" ))


#Step 2 : Pooling


classifier.add(MaxPooling2D(pool_size = (2,2)))

#part3 improving accuracy on test set:
#we need to change input shape, our CNN should expect what dimension as the input of our image. the input is going to be the pooled feature map of previous layer. so it is not image. so we dont need to include the input shape. so we only need 1- number of feature detectors, 2- dimension of feature detectors ,3 activation function

classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation="relu" ))

classifier.add(MaxPooling2D(pool_size = (2,2)))

#step 3: Flattening >> taking all feature map and put them into one single vector

classifier.add(Flatten())

#Since we already have our input layer we only need to create a hidden layer
#Step4 : Full connection

classifier.add(Dense(128, activation= 'relu'))

#now we add output layer, if we had outcome with more than two categories: softmax
classifier.add(Dense(1, activation= 'sigmoid'))

#now to compile the whole thing: we need to choose a stochastic gradient descent algorithm, a loss function and performance metrics
#if we had more than 2 categories, loss function would be categorical_crossentropy

classifier.compile(optimizer ='adam', loss= 'binary_crossentropy', metrics=['accuracy'])


#part2
#image preprocessing: we will fit our CNN that we just built to all our images
#we use a shortcut in Keras, its for image augmentation and consist of preprocessing your images to prevent overfitting
#in Bbrowser type Keras Documentation- preprocessing, we use the code flow_from_directory, instead of directory we put our dataset.
#with this code, besides preprocessing and augmentation we even can fit our CNN that we just built on our images.
#fit_generator method not only fit the CNN to the training set, but it will also test at the same time its performance on some new operations which are gonna be the observations of our test set

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#here we fit out CNN to training set and test it on test set
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


#so far the accuracy on training set and test set were somewhat different. therefore inorder to improve the accuracy on test set:
#make our CNN deeper : either by adding another CNN or add another fully connected layer
#in this toturial we add another convolutional layer.
#we add the 2nd layer of CNN right after maxpooling part. afterwards we need to add another maxpooling


#if you still want to improve more either add more layer, and increase the feature detector, or even if you want to improve further increase the target size, because that means that we have more number of pixels in one image therefore more info


#Part 4 making new predictions

#start with 1st image cat_or_dog_1 (which is a dog), we want to use a function from numpy to preprocess the image that we want to load, so that it can be accepted by the prediction method that we want to use for single prediction

import numpy as np

# now we need to import the second module which is the  preprocessing image module from Keras

from keras.preprocessing import image
test_image = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg", target_size = (64,64))

#arguments of load are path of image and target size which should be as same as target size of training set

#our image is colored image so it has another dimension like training set (input shape): 64,64,3
#so we need to create another dimension

test_image= image.img_to_array(test_image)

#we need to add another dimension to our 3 dimensional array, this dimension corresponds to the batch, predict function only accepts input in a batch even batch only has one element, so modify our variable again

test_image=np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

#1 corresponds to cat or dog? use class indexes

training_set.class_indices
if result [0][0]==1:
    prediction = "dog"
else:
    prediction=cat

#This is just to test if I can branch and commit to branch