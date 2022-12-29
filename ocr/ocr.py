import numpy as np
import cv2
import idx2numpy
import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import operator
from operator import itemgetter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from KNN import KNN
import time


# Set image path

fileName = "mrrm9.png"

# Read input image:
inputImage = cv2.imread(fileName)
inputCopy = inputImage.copy()

# Convert BGR to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)


# adaptive thereshould parameter to separate the foreground image from the background
windowSize = 43
windowConstant = -1
# Apply the threshold:
binaryImage = cv2.adaptiveThreshold(
    grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, windowSize, windowConstant)


# Perform an area filter on the binary blobs:
componentsNumber, labeledImage, componentStats, componentCentroids = cv2.connectedComponentsWithStats(
    binaryImage, connectivity=4)

# Set the minimum pixels for the area filter:
minArea = 43

# Get the indices/labels of the remaining components based on the area stat
# (skip the background component at index 0)
remainingComponentLabels = [i for i in range(
    1, componentsNumber) if componentStats[i][4] >= minArea]

# Filter the labeled pixels based on the remaining labels,
# assign pixel intensity to 255 (uint8) for the remaining pixels
filteredImage = np.where(
    np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')


# Set kernel (structuring element) size:
kernelSize = 3

# Set operation iterations:
opIterations = 1

# Get the structuring element:
maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

# Perform closing:
closingImage = cv2.morphologyEx(
    filteredImage, cv2.MORPH_CLOSE, maxKernel, None, None, opIterations, cv2.BORDER_REFLECT101)


# Get each bounding box
# Find the big contours/blobs on the filtered image:
contours, hierarchy = cv2.findContours(
    closingImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

contours_poly = [None] * len(contours)
# The Bounding Rectangles will be stored here:
boundRect = []

# Alright, just look for the outer bounding boxes:
for i, c in enumerate(contours):

    if hierarchy[0][i][3] == -1:
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect.append(cv2.boundingRect(contours_poly[i]))


# Draw the bounding boxes on the (copied) input image:
for i in range(len(boundRect)):
    color = (0, 255, 0)
    cv2.rectangle(inputCopy, (int(boundRect[i][0]), int(boundRect[i][1])),
                  (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)


def fillWithZeros(arr):
    return cv2.resize(arr, (28, 28))


# loading train images and test labels
# trainimage is 60000 ,28 , 28 shape
imagetrain = idx2numpy.convert_from_file('./dataset/train-images.idx3-ubyte')
imagetrainlabel = idx2numpy.convert_from_file(
    './dataset/train-labels.idx1-ubyte')

# loding test images and test labels
imagetest = idx2numpy.convert_from_file('./dataset/test-images.idx3-ubyte')
imagetestlable = idx2numpy.convert_from_file(
    './dataset/test-labels.idx1-ubyte')


def squeze_data(imagetrain, imagetrainlabel, imagetest, imagetestlable):
    # take the image part flatten it and return
    return imagetrain, imagetrainlabel, imagetest, imagetestlable
    # return imagetrain , imagetrainlabel , imagetest , imagetestlable


def test(X_train, y_train, X_test):
    clf = KNN(k=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions


X_train, y_train, X_test, Y_test = squeze_data(
    imagetrain, imagetrainlabel, imagetest, imagetestlable)


arr = []
# Crop the characters:
for i in range(len(boundRect)):
    # Get the roi for each bounding rectangle:
    x, y, w, h = boundRect[i]
    # Crop the roi:
    croppedImg = closingImage[y:y + h, x:x + w]
    reshappedimage = fillWithZeros(croppedImg)
    arr.append(test(X_train, y_train, reshappedimage))

print(arr)
