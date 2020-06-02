# import the necessary packages
from imutils import face_utils
from sklearn import svm
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import VotingClassifier
import numpy as np
import imutils
import dlib
import cv2
import sys
import os
import math
import pandas as pd
import pickle

model_file = "Finalized_model6epochs.sav"
training_directory = "Original Dataset"

emotion_labels = [0]*8

labels_dict = { '0': 'Neutral', 
				'1': 'Anger', 
				'2': 'Contempt', 
				'3': 'Disgust', 
				'4': 'Fear', 
				'5': 'Happy', 
				'6': 'Sadness', 
				'7': 'Surprise'}

# For total training images
count = 0

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
		return rect, shape

# flag 0 = For Training Images
# flag 1 = For Runtime Images
def extractLandmarks(frame, flag):

	global featureMatrix
	
	# Detecting facial landmarks & store the 68 2D-coordinates in landmarks(numpy array)
	rect, landmarks = facialDetector(frame)
	
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

	# For Training Images we store landmarks in text document
	if flag==0:
		f = open("Training Features.txt", "a")
		# Converting each float value in lt list to string to write it in the text file
		f.write(','.join(str(number) for number in lt))
		f.write("\n")
		f.close()

	# For Runtime Images insert landmarks in the numpy array
	else:
		featureMatrix = np.insert(featureMatrix, 0, np.array([lt]), axis=0)

	return rect

def extractLabels(label_file):

	global labels
	global count
	global emotion_labels

	f = open(label_file)
			
	label = f.read(4)			# reading first 4 characters
	label = label.lstrip()		# deleting left whitespaces
	
	if label != '':
		emotion_labels[int(label)] += 1
	
	labels = np.append(labels, label)	
	count += 1

	f.close()
	
def extractFeatures(directory):

	for file in os.listdir(directory):

		location = directory + '/' + file

		# If it is an image file
		if "png" in file:
			# Reading the image from above path
			image = cv2.imread(location)
			output = extractLandmarks(image, 0)

		# If it is a text i.e emotion file
		elif "txt" in file:
			extractLabels(location)
		else:
			
			if os.path.isdir(location):
				# print (file)
				if "S" in file:
					print (file)
				extractFeatures(location)

def readLandmarks(file):

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
	print ("Landmarks Read")	

def trainModel():

	global featureMatrix
	global labels
	global model_file

	no_of_iterations = 6

	print ()
	print ("Training Epochs :")

	# The possible values of tuning parameters of svm
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.1, 10, 25, 50, 100, 1000]}
                   ]

	# KFold cross-validation Method implemented with dataset divided into 10 folds
	kf = StratifiedKFold( n_splits=no_of_iterations )
	i = 1

	for train_index, test_index in kf.split(featureMatrix, labels):

		print (i)
		print ("Train: ", len(train_index), " images\nTest: ", len(test_index), " images")

		# Splitting the featureMatrix i.e. 9 folds contains training images & 1 fold contains testing images
		train_feature, test_feature = featureMatrix[train_index], featureMatrix[test_index]

		# Splitting the labels i.e. 9 folds contains training labels & 1 fold contains testing labels
		train_labels, test_labels = labels[train_index], labels[test_index]

		# Selecting the best combination of tuning paramters
		grid_search = GridSearchCV(svm.SVC(), tuned_parameters, cv=3, iid='False')

		# Training the train_dataset on tuned parameters
		grid_search.fit(train_feature, train_labels)

		i += 1
		
	pickle.dump(grid_search, open(model_file, 'wb'))
	# pickle.dump(model, open(file_name, 'wb'))

def countImagesPerEmotion():

	global emotion_labels

	extractFeatures(training_directory + '/' + 'Emotion Labels')
	
	f = open("Images Per Emotion.txt","w")

	i = 0
	for l in emotion_labels:
		print (labels_dict[str(i)], l)	
		f.write(labels_dict[str(i)] + ' ' +  str(l))
		f.write('\n')
		i += 1	

	f.close()

def ensemble(test_feature, loaded_model1, loaded_model2, loaded_model3, loaded_model4, loaded_model5, loaded_model6	):

	global featureMatrix
	global labels


	pred1 = loaded_model1.predict(test_feature)
	pred2 = loaded_model2.predict(test_feature)
	pred3 = loaded_model3.predict(test_feature)
	pred4 = loaded_model4.predict(test_feature)
	pred5 = loaded_model5.predict(test_feature)
	pred6 = loaded_model6.predict(test_feature)

	estimators = []
	estimators.append(pred1[0])
	estimators.append(pred2[0])
	estimators.append(pred3[0])
	estimators.append(pred4[0])
	estimators.append(pred5[0])
	estimators.append(pred6[0])

	return estimators
	# return Counter(estimators).most_common(1)[0][0]
	
	# estimators.append(('model1',loaded_model1))
	# estimators.append(('model2',loaded_model2))
	# estimators.append(('model3',loaded_model3))
	# estimators.append(('model4',loaded_model4))
	# estimators.append(('model5',loaded_model5))
	# estimators.append(('model6',loaded_model6))
	
	# ensemble = VotingClassifier(estimators)
	
	# print ('Fitting starting....')
	# ensemble.fit(featureMatrix, labels)

	# print ('Ensembled Model fitted')

	# pickle.dump(ensemble, open('Ensembled_model_file.sav', 'wb'))

def main():

	global featureMatrix
	global labels
	global count
	global emotion_labels
	global model_file

	# If model is not trained then train it and save it in the file
	if not os.path.exists(model_file):
		
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

		trainModel() 

	count = 10708

	loaded_model1 = pickle.load(open('Finalized_model2epochs.sav', 'rb'))
	loaded_model2 = pickle.load(open('Finalized_model3epochs.sav', 'rb'))
	loaded_model3 = pickle.load(open('Finalized_model5epochs.sav', 'rb'))
	loaded_model4 = pickle.load(open('Finalized_model6epochs.sav', 'rb'))
	loaded_model5 = pickle.load(open('Finalized_model7epochs.sav', 'rb'))
	loaded_model6 = pickle.load(open('Finalized_model8epochs.sav', 'rb'))

	print ('Models Loaded')

	print ("Training Done on {} Images".format(count))

	cap = cv2.VideoCapture (0)
	
	while True:
		
		try:
			featureMatrix = np.empty( (0, 68) )
			ret, frame = cap.read ()	

			rect = extractLandmarks(frame, 1)
			test_feature = featureMatrix

			predicted_label = []
			# Predicting the labels for testing images
			# predicted_label = ensemble(test_feature)
			# predicted_label = loaded_model.predict1(test_feature)
			predicted_label = ensemble(test_feature, loaded_model1, loaded_model2, loaded_model3, loaded_model4, loaded_model5, loaded_model6)

			x = rect.left() + 70
			y = rect.top() - 25
			w = rect.right() + 100
			h = rect.bottom() + 50

			freq = [0]*8

			for predicted in predicted_label:
				try :
					freq[int(predicted)] += 1
				except ValueError:
					print (predicted)

			# print (freq)

			prob = [0]*8

			for i in range(0,8):
				prob[i] = (freq[i]/6)*100

			cv2.rectangle(frame, (x, y), (w, h), (0,0,0), 2)
			
			j = 90
			for i in range(0,8):
				
				# print (labels_dict[str(i)] + ": " + str(prob[i]))
				maximum_prob = max(prob)

				string = str(prob[i]).split(".")
				probability = string[0] + "%"

				if prob[i] == maximum_prob:
					cv2.putText(frame, labels_dict[str(i)] + ": " + probability, (30, j), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				else:
					cv2.putText(frame, labels_dict[str(i)] + ": " + probability, (30, j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
				j += 30
			
			cv2.imshow ('WebCam', frame)
			# cv2.resize('WebCam', 700, 700)

			if cv2.waitKey (1) == 27: #27 is the Escape Key
				sys.exit()

		except TypeError:
			pass


	# When everything done, release the capture
	cap.release ()
	cv2.destroyAllWindows ()


if __name__ == '__main__':
	main()