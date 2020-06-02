# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import imutils
import dlib
import cv2
import sys
import os
import math
import pandas as pd

t_directory = "Dataset Small/Emotion Labels"

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
		# cv2.imshow("Image", output)
		# cv2.waitKey(0)
		return shape

def extractFeatures(directory):

	for file in os.listdir(directory):

		# If we found a text doc in the directory, then extract label & landmarks
		if "txt" in file:
			
			global row
			global featureMatrix
			global labels
			
			# To extract the labels from the text documents present in directories
			file_dir = directory + "/" + file
			f = open(file_dir)
			
			label = f.read(4)			# reading first 4 characters
			label = label.lstrip()		# deleting left whitespaces
			labels = np.append(labels, label)	
			
			# To extract the images from the directories
			image_path = file_dir.replace("_emotion.txt",".PNG")
			image_path = image_path.replace("Emotion Labels","Images")
			
			# Reading the image from above path
			image = cv2.imread(image_path)
			
			# Detecting facial landmarks & store the 68 2D-coordinates in landmarks(numpy array)
			landmarks = facialDetector(image)
			
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

			# Appending the "lt" list which stores the face's landmark's location in numpy array
			featureMatrix = np.append( featureMatrix, np.array([lt]), axis=0 )

		else:
			# Printing the directories
			if "S" in file:
				print (file)
			
			# If no text doc, then search the subdirectory
			extractFeatures(directory + "/" + file)

def main():

	global featureMatrix
	global labels

	print ("Training Directories: ")
	# Extracting labels & landmarks for all the dataset
	extractFeatures(t_directory)

	# print ()
	# print ("Total training images: ", train_feature.shape[0])

	# The possible values of tuning parameters of svm
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.1, 10, 25, 50, 100, 1000]}
                   ]



	train_feature, test_feature, train_labels, test_labels = train_test_split(featureMatrix, labels, test_size=0.2)

	# Selecting the best combination of tuning paramters
	grid_search = GridSearchCV(svm.SVC(), tuned_parameters, cv=3, iid='False')
		
	# Training the train_dataset on tuned parameters
	grid_search.fit(train_feature, train_labels)
	
	# Predciting the labels for testing images
	predicted_labels = grid_search.predict(test_feature)

		
	# Comparing & calculating the accuracy of this epoch
	accuracy = accuracy_score(test_labels, predicted_labels)

	# print ()
	# print ("Best Possible Combination: ", grid_search.best_params_)

	# Showing all the ranks of possible combinations in pandas Dataframe
	# data = { 'Rank' : grid_search.cv_results_['rank_test_score'], 'Params' : grid_search.cv_results_['params'] }
	# df = pd.DataFrame(data)
	# print (df)
	
	print ("\nNo. of Training Images : {}\nNo. of Testing Images : {}".format(len(train_feature), len(test_feature)))

	# Displaying the average accuracy	
	print ("\nFINISHED classifying. \nAverage Accuracy score(%) :")
	print ( accuracy )

	print ("\nTest Dataset with maximum accuracy:")
	# Showing the comparison b/w Test & Predicted Labels in pandas Dataframe
	data_labels = { 'Test_Labels' : test_labels, 'Predicted_Labels' : predicted_labels }
	df_labels = pd.DataFrame(data_labels)
	# Condition required to match the columns
	df_labels['Match'] = np.where( df_labels['Test_Labels'] == df_labels['Predicted_Labels'], True, False)
	# Replacing '1', '2', etc numerical labels with actual emotion labels 'Surprise', 'Anger', etc 
	df_labels = df_labels.replace( { "Test_Labels" : labels_dict, "Predicted_Labels" : labels_dict} )

	print ()
	# print (df_labels)
	print ("\nNo. of Matches: ")
	print ( len( df_labels[df_labels['Match'] == True] ) , "out of" , len( df_labels.index ) )
	
if __name__ == '__main__':
	main()