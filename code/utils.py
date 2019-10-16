import cv2
import numpy
import time
from sklearn import neighbors, svm, cluster
from sklearn.cluster import AgglomerativeClustering
from classifiers import KNN_classifier

def readImages(imagesRaw, newSize):
	new_list = []
	for i in range(len(imagesRaw)):
		currentResizedImage = imresize(imagesRaw[i], newSize)
		new_list.append(currentResizedImage)
	return new_list

def readImagesNonNormalized(imagesRaw, newSize):
	new_list = []
	for i in range(len(imagesRaw)):
		currentResizedImage = imresizeNonNormalized(imagesRaw[i], newSize)
		new_list.append(currentResizedImage)
	return new_list

def convert(original_list, newSize):
	# takes the list of images, converts them to 2D matrices, and then converts that to a vector for the classifier
	new_list = readImages(original_list, newSize)
	data = numpy.array(new_list)
	n, nx, ny = data.shape
	final_data = data.reshape((n, nx * ny))
	return final_data

def imresize(input_image, target_size):
	# resizes the input image to a new image of size [target_size, target_size]. normalizes the output image
	# to be zero-mean, and in the [-1, 1] range.
	dim = (target_size, target_size)
	img = cv2.resize(input_image, dim)

	# Normalize image from [-255, 255]
	output_image = cv2.normalize(img, None, -1, 1, cv2.NORM_MINMAX)
	return output_image

def imresizeNonNormalized(input_image, target_size):
	# resizes the input image to a new image of size [target_size, target_size]. normalizes the output image
	# to be zero-mean, and in the [-1, 1] range.
	dim = (target_size, target_size)
	output_image = cv2.resize(input_image, dim)

	return output_image

def reportAccuracy(true_labels, predicted_labels, label_dict):
	# generates and returns the accuracy of a model
	# true_labels is a n x 1 cell array, where each entry is an integer
	# and n is the size of the testing set.
	# predicted_labels is a n x 1 cell array, where each entry is an 
	# integer, and n is the size of the testing set. these labels 
	# were produced by your system
	# label_dict is a 15x1 cell array where each entry is a string
	# containing the name of that category
	# accuracy is a scalar, defined in the spec (in %)
	numCorrectPredictions = 0
	numPredictions = len(predicted_labels) 

	for i in range(numPredictions):
		if predicted_labels[i] == true_labels[i]:
			numCorrectPredictions = numCorrectPredictions + 1

	accuracy = numCorrectPredictions / numPredictions
	return accuracy

def buildDict(train_images, dict_size, feature_type, clustering_type):
	# this function will sample descrichromeptors from the training images,
	# cluster them, and then return the cluster centers.

	# train_images is a n x 1 array of images
	# dict_size is the size of the vocabulary,
	# feature_type is a string specifying the type of feature that we are interested in.
	# Valid values are "sift", "surf" and "orb"
	# clustering_type is one of "kmeans" or "hierarchical"

	# the output 'vocabulary' should be dict_size x d, where d is the 
	# dimention of the feature. each row is a cluster centroid / visual word.
	# --- Sampling 
	sift = cv2.xfeatures2d.SIFT_create()
	surf = cv2.xfeatures2d.SURF_create()
	orb = cv2.ORB_create() # K Means Clustering only works when detectAndCompute() is run with raw train_images

	processed_training_images = readImagesNonNormalized(train_images, 32)
	all_features = []
	for i in range(len(processed_training_images)):
		keypoints_sift, descriptors = sift.detectAndCompute(processed_training_images[i], None)
		feature = (keypoints_sift, descriptors)
		if descriptors is None:
			continue
		else:
			for matrix in descriptors:
				all_features.append(matrix)
	
	print(len(all_features))
	# print(all_features)

	# --- K Means Clustering 
	# kmeans = cluster.KMeans(dict_size)
	# kmeans.fit(all_features)

	# --- Agglomerative Clustering
	# npmy = numpy.array(all_features) # orb only, need to feed this into  AgglomerativeClustering().fit
	# npmy = npmy.reshape(-1, 1) # orb only
	clustering = AgglomerativeClustering(n_clusters=dict_size).fit(all_features)
	return clustering

def computeBow(image, vocabulary, feature_type):
	# extracts features from the image, and returns a BOW representation using a vocabulary
	# image is 2D array
	# vocabulary is an array of size dict_size x d
	# feature type is a string (from "sift", "surf", "orb") specifying the feature
	# used to create the vocabulary

	# BOW is the new image representation, a normalized histogram
	return Bow

def tinyImages(train_features, test_features, train_labels, test_labels, label_dict):
	# train_features is a nx1 array of images
	# test_features is a nx1 array of images
	# train_labels is a nx1 array of integers, containing the label values
	# test_labels is a nx1 array of integers, containing the label values
	# label_dict is a 15x1 array of strings, containing the names of the labels
	# classResult is a 18x1 array, containing accuracies and runtimes
	classResult = []

	for imageSize in [8, 16, 32]:
		for numNeighbors in [1, 3, 6]:
			start_time = time.time()
			converted_training_set = convert(train_features, imageSize)
			converted_test_set = convert(test_features, imageSize)
			predictions = KNN_classifier(converted_training_set, train_labels, converted_test_set, 1)
			timeTaken = time.time() - start_time
			accuracy = reportAccuracy(test_labels, predictions, None)
			classResult.append(accuracy)
			classResult.append(timeTaken)

	print(classResult)
	return classResult
	
