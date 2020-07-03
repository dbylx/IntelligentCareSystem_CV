# -*- coding: utf-8 -*-
'''
test mnist
'''

# import necessary libraries
import numpy as np
import cv2
from keras.models import load_model
from keras.datasets import mnist

# global variables
model_path = 'models/mnist.hdf5'

model = load_model(model_path)

# grab the MNIST dataset
print('[INFO] 下载数据集...')
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# matrix shape should be: num_samples x rows x columns x depth
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(5,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # extract the image from the testData
    # image = (testData[i] * 255).astype('uint8')

    # merge the channels into one image, i.e. let the image
    # has 3 channels, otherwise our colorful predicted number
    # won't appear
    image = cv2.merge([testData[i]] * 3)

    # resize the image from 28x28 to a 96x96 image so that we can
    # better see it
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

    # show the image and prediction
    cv2.putText(image, str(prediction[0]), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    print('[INFO] Predicted: {}, Actual:{}'.format(prediction[0],
                                                   testLabels[i]))
    cv2.imshow('Digit', image)
    cv2.waitKey(0)

