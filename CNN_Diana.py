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

#image preprocessing: we will fit our CNN that we just built to all our images