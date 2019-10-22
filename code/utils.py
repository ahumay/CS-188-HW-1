import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

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
    data = np.array(new_list)
    n, nx, ny = data.shape
    final_data = data.reshape((n, nx * ny))
    return final_data

def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, 0)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, 0)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(train_features, train_labels)
    predicted_categories = neigh.predict(test_features);
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    return predicted_categories


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


def reportAccuracy(true_labels, predicted_labels):
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
    return accuracy * 100

def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.
    # --- Sampling 

    print(feature_type, clustering_type)
    allDescriptors = []
    if feature_type == 'sift':
        processed_training_images = readImagesNonNormalized(train_images, 16)
        sift = cv2.xfeatures2d.SIFT_create()
        for i in range(len(processed_training_images)):
            keypoints_sift, descriptors = sift.detectAndCompute(processed_training_images[i], None)
            if descriptors is None:
                continue
            else:
                for matrix in descriptors:
                    allDescriptors.append(matrix)

    elif feature_type == 'surf':
        processed_training_images = readImagesNonNormalized(train_images, 32)
        surf = cv2.xfeatures2d.SURF_create() 
        for i in range(len(processed_training_images)):
            keypoints_sift, descriptors = surf.detectAndCompute(processed_training_images[i], None)
            if descriptors is None:
                continue
            else:
                for matrix in descriptors:
                    allDescriptors.append(matrix)

    elif feature_type == 'orb':
        return None
        processed_training_images = readImages(train_images, 128)
        orb = cv2.ORB_create() 
        # for i in range(len(train_images)):
        #     keypoints_sift, descriptors = orb.detectAndCompute(train_images[i], None)
        #     if descriptors is None:
        #         continue
        #     else:
        #         for matrix in descriptors:
        #             allDescriptors.append(matrix)
        for i in range(len(processed_training_images)):
            kp = orb.detect(processed_training_images[i], None)
            kp, des = orb.compute(processed_training_images[i], kp)
            allDescriptors.append(des)
            if des is None:
                continue
            else:
                for matrix in des:
                    allDescriptors.append(matrix)

        # if clustering_type == 'hierarchical':
        # npmy = np.array(allDescriptors) # Orb only, need to feed this into  AgglomerativeClustering().fit
        # npmy = npmy.reshape(-1, 1) # Orb only

    else:
        print("Error. Check feature type")
        return None
        
    # --- Orb the way TA wanted it
    # allDescriptors = []
    # for i in range(len(train_images)):
    #     kp = orb.detect(train_images[i], None)
    #     kp, des = orb.compute(train_images[i], kp)
    #     allDescriptors.append(des)
    #     if des is None:
    #         continue
    #     else:
    #         for matrix in des:
    #             allDescriptors.append(matrix)
    # data = np.array(allDescriptors)
    # final_data = data.flatten()
    # print("len(final_data):" + str(len(final_data)))

    if clustering_type == 'kmeans':
        clustering = cluster.KMeans(dict_size)
        clustering.fit(allDescriptors)
        centers = np.array(clustering.cluster_centers_)

    elif clustering_type == 'hierarchical':
        clustering = AgglomerativeClustering(n_clusters=dict_size).fit(allDescriptors)
        centersDict = {}

        # Calculate sum of all descriptors sharing the same label
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] not in centersDict:
                centersDict[clustering.labels_[i]] = (allDescriptors[i], 1)
            else:
                (curAccumulatedDescriptor, numberOfDesciptorsRead) = centersDict[clustering.labels_[i]]
                for idx, val in enumerate(allDescriptors[i]):
                    curAccumulatedDescriptor[idx] = val+curAccumulatedDescriptor[idx]
                centersDict[clustering.labels_[i]] = (curAccumulatedDescriptor, numberOfDesciptorsRead + 1)
        
        # Calculate the centroids from agg clustering by averaging all points
        # corresponding to a certain label
        for key in centersDict:
            (curAccumulatedDescriptor, numberOfDescriptorsRead) = centersDict[key]
            averageDescriptor = curAccumulatedDescriptor / numberOfDescriptorsRead
            centersDict[key] = (averageDescriptor, numberOfDescriptorsRead)

        # Create a numpy array of agglomerative clustering centroids
        agg_centroids = []
        for key in centersDict.keys():
            agg_centroids.append(centersDict[key])
        centers = np.array(agg_centroids)

    else:
        print("Error. Check clustering type")
        return None

    return centers

def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    processed_image = imresizeNonNormalized(image, 32)
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create() 
    orb = cv2.ORB_create()
    all_descriptors = []
    keypoints_sift, descriptors = sift.detectAndCompute(processed_image, None)
    if descriptors is None:
        print("NONE")
        pass
    else:
        for desc in descriptors:
            all_descriptors.append(desc)

    print("vocab", len(vocabulary))
    print("labels", len([x for x in range(1, 21)]))
    all_descriptors = np.array(all_descriptors)
    print(all_descriptors)

    predictions = KNN_classifier(vocabulary, [x for x in range(1, 21)], all_descriptors, 3)
    print(predictions)

    pred_dict = {}
    for pred in predictions:
        if pred not in pred_dict:
            pred_dict[pred] = 1
        else:
            pred_dict[pred] += 1
    print("histogram dict: ", pred_dict)

    return Bow
    
def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    classResult = []

    for imageSize in [8, 16, 32]:
        for numNeighbors in [1, 3, 6]:
            start_time = time.time()
            converted_training_set = convert(train_features, imageSize)
            converted_test_set = convert(test_features, imageSize)
            predictions = KNN_classifier(converted_training_set, train_labels, converted_test_set, numNeighbors)
            timeTaken = time.time() - start_time
            accuracy = reportAccuracy(test_labels, predictions)
            classResult.append(accuracy)
            classResult.append(timeTaken)

    print(classResult)
    return classResult
    