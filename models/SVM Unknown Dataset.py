# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import imutils
import dlib
import cv2
import sys
import os
import math
import pandas as pd

training_directory = "Dataset Small"
testing_directory = "images"

labels_dict = { '0': 'Neutral', 
				'1': 'Anger', 
				'2': 'Contempt', 
				'3': 'Disgust', 
				'4': 'Fear', 
				'5': 'Happy', 
				'6': 'Sadness', 
				'7': 'Surprise'}

# Maintaining a numpy array for storing labels
labels = np.empty((0, 0))
# Maintaining a numpy array for storing landmarks(feature matrices) of each image
featureMatrix = np.empty( (0, 68) )

detector = 	dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def facialDetector(image):	
	# load the input image, resize it, and convert it to grayscale
	# image = cv2.imread(args["image"])
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# visualize all facial landmarks with a transparent overlay
		output = face_utils.visualize_facial_landmarks(image, shape)
		return output, shape

def extractLandmarks(image_file):

	global featureMatrix

	# Reading the image from above path
	image = cv2.imread(image_file)
	
	# Detecting facial landmarks & store the 68 2D-coordinates in landmarks(numpy array)
	output, landmarks = facialDetector(image)
	
	lt = []

	# Converting the 68 (x, y) 2D-coordinates to 68 1D-coordinates by applying (x*x + y*y) ^ (1/2) to make it unique
	for coordinates in landmarks:
		temp = math.sqrt ( math.pow(coordinates[0],2) + math.pow(coordinates[1],2) )

		# if we encounter (y, x) for which (x, y) has already been calculated
		if temp in lt:

			# if (x<y) then postive value otherwise negative value
			if coordinates[0] < coordinates[1]:
				lt.append(temp)
			else:
				lt.append(-temp)
		else:
			lt.append(temp)

	# Don't want to store the landmarks of testing images to text document
	if testing_directory not in image_file:
		f = open("Training Features.txt", "a")
		# Converting each float value in lt list to string to write it in the text file
		f.write(','.join(str(number) for number in lt))
		f.write("\n")
		f.close()

	# Incase of testing images, store the landmarks in featureMatrix numpy array 
	else:
		# Inserting the landmarks in the numpy array
		featureMatrix = np.insert(featureMatrix, 0, np.array([lt]), axis=0)

	return output

def extractLabels(label_file):

	global labels

	f = open(label_file)
			
	label = f.read(4)			# reading first 4 characters
	label = label.lstrip()		# deleting left whitespaces
	labels = np.append(labels, label)	

	f.close()
	
def extractFeatures(directory):

	for file in os.listdir(directory):

		location = directory + '/' + file

		# If it is an image file
		if "png" in file:
			output = extractLandmarks(location)
		# If it is a text i.e emotion file
		elif "txt" in file:
			extractLabels(location)
		else:
			
			if os.path.isdir(location):
				if "S" in file:
					print (file)
				extractFeatures(location)

def readLandmarks(file):

	print ("Reading landmarks....")
	global featureMatrix

	f = open(file)
	line = f.readline()

	while line:

		# Splitting each line by ',' character to get landmarks for that image
		temp = line.split(',')
		# Converting the string value to float value for every elememt in the list
		temp = list(map(float, temp))
		# Appending the "lt" list which stores the face's landmark's location in numpy array
		featureMatrix = np.append( featureMatrix, np.array([temp]), axis=0 )

		line = f.readline()

	f.close()

def main():

	global featureMatrix
	global labels

	print ("Training Directories: ")
	
	# Extracting landmarks only when the text document (containing the landmarks for every face in training dataset) does not exist
	if not os.path.exists("Training Features.txt"):
		print ("Images")
		extractFeatures(training_directory + '/' + 'Images')
	
	print ("Emotion Labels")
	# Extracting Labels for the training dataset
	extractFeatures(training_directory + '/' + 'Emotion Labels')

	# Reading the landmarks from the text document
	readLandmarks("Training Features.txt")

	# The possible values of tuning parameters of svm
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.1, 10, 25, 50, 100, 1000]}
                   ]

	no_of_iterations = 10

	print ()
	print ("Training Epochs :")

	# KFold cross-validation Method implemented with dataset divided into 10 folds
	kf = StratifiedKFold( n_splits=no_of_iterations )
	i = 1

	for train_index, test_index in kf.split(featureMatrix, labels):

		# Splitting the featureMatrix i.e. 9 folds contains training images & 1 fold contains testing images
		train_feature, test_feature = featureMatrix[train_index], featureMatrix[test_index]

		# Splitting the labels i.e. 9 folds contains training labels & 1 fold contains testing labels
		train_labels, test_labels = labels[train_index], labels[test_index]
		
		# Selecting the best combination of tuning paramters
		grid_search = GridSearchCV(svm.SVC(), tuned_parameters, cv=3, iid='False')
		
		# Training the train_dataset on tuned parameters
		grid_search.fit(train_feature, train_labels)
		
		print (i)
		i += 1
	
	featureMatrix = np.empty( (0, 68) )

	for img in os.listdir(testing_directory):
		
		# Extracting landmarks for testing images
		output = extractLandmarks(testing_directory + '/' + img)
		test_feature = featureMatrix

		# Predicting the labels for testing images
		predicted_label = grid_search.predict(test_feature)

		image = cv2.imread(testing_directory + '/' + img)
		image = imutils.resize(image, width=500)

		# Putting text of predicted label on the image
		cv2.putText(image, labels_dict[predicted_label[0]], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# "output" contains the image having outlines of landmarks on the face & "image" contains the original image
		# Combining both images with equal intensities
		combined_image = cv2.addWeighted(image, 0.5, output, 0.5, 0)
		
		cv2.imshow("Image", combined_image)
		cv2.waitKey(0)


if __name__ == '__main__':
	main()