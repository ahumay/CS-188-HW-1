from utils import *
import argparse

parser = argparse.ArgumentParser(description='CS188.2 - Fall 19 - Homework 1')
parser.add_argument("--tiny", "-t", type=bool, default=True, help='run Tiny Images')
parser.add_argument("--create-path", "-cp", type=bool, default=True, help='create the Results directory')
parser.parse_args()
args = parser.parse_args()

# The argument is included as an idea for debugging, with a few examples in the main. Feel free to modify it or add arguments.
# You are also welcome to disregard this entirely

#############################################################################################################################
# This file contains the main. All the functions you need to write are included in utils. You also need to edit the main.
# The main just gets you started with the data and highlights the high level structure of the project.
# You are free to modify it as you wish - the modifications you are required to make have been marked but you are free to make
# others.
# What you cannot modify is the number of files you have to save or their names. All the save calls are done for you, you
# just need to specify the right data.
#############################################################################################################################


if __name__ == "__main__":
    
    if args.create_path:
        # To save accuracies, runtimes, voabularies, ...
        if not os.path.exists('Results'):
            os.mkdir('Results') 
        SAVEPATH = 'Results/'
    
    # Load data, the function is written for you in utils
    train_images, test_images, train_labels, test_labels = load_data()

    # UNCOMMENT THIS LATER
    # if args.tiny:
    #     # You have to write the tinyImages function
    #     tinyRes = tinyImages(train_images, test_images, train_labels, test_labels)
    
    #     # Split accuracies and runtimes for saving  
    #     for element in tinyRes[::2]:
    #         # Check that every second element is an accuracy in reasonable bounds
    #         assert (7 < element and element < 20)
    #     acc = np.asarray(tinyRes[::2])
    #     runtime = np.asarray(tinyRes[1::2])
    
    #     # Save results
    #     np.save(SAVEPATH + 'tiny_acc.npy', acc)
    #     np.save(SAVEPATH + 'tiny_time.npy', runtime)

    # Create vocabularies, and save them in the result directory
    # You need to write the buildDict function
    vocabularies = []
    vocab_idx = [] # If you have doubts on which index is mapped to which vocabulary, this is referenced here
    # e.g vocab_idx[i] will tell you which algorithms/neighbors were used to compute vocabulary i
    # This isn't used in the rest of the code so you can feel free to ignore it



    ### -------------------------------- Part 1 -------------------------------------------###
    ### -----------------------------------------------------------------------------------###
    ### -----------------------------------------------------------------------------------###

    # Build multiple dictionaries using varying methods
    # COMMENTED SO WE DON'T RECOMPUTE DESCRIPTORS
    '''
    for feature in ['sift', 'surf']:
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                print("Building vocabulary...")
                vocabulary = buildDict(train_images, dict_size, feature, algo)
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                np.save(SAVEPATH + filename, np.asarray(vocabulary))
                vocabularies.append(vocabulary) # A list of vocabularies (which are 2D arrays)
                vocab_idx.append(filename.split('.')[0]) # Save the map from index to vocabulary
                print("Saved vocabulary using feature_extractor={}, cluster_alg={}, and dict_size={}. FILEPATH: {}".format(
                    feature, algo, dict_size, str(SAVEPATH+filename)))
    '''
    # Compute the Bow representation for the training and testing sets
    test_rep = [] # To store a set of BOW representations for the test images (given a vocabulary)
    train_rep = [] # To store a set of BOW representations for the train images (given a vocabulary)
    features = ['sift'] * 4 + ['surf'] * 4 + ['orb'] * 4 # Order in which features were used 
    # for vocabulary generation


    ### -------------------------------- Part 2 -------------------------------------------###
    ### -----------------------------------------------------------------------------------###
    ### -----------------------------------------------------------------------------------###

    # Load saved vocabularies (NEED TO ADD ORB)
    print("Loading vocabulary...")
    for feature in ['sift', 'surf']:
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                vocab = np.load(SAVEPATH + filename)
                vocabularies.append(vocab)
    print("Vocabulary loaded")

    
    # COMMENTED SO WE DON'T RECOMPUTE BOW REPRESENTATION OF IMAGES
    '''
    # Compute BOW representation of each image, for each vocabulary
    print("Computing BOW representation of each image, for each vocabulary")
    for i, vocab in enumerate(vocabularies):

        for image in train_images: # Compute the BOW representation of the training set
            rep = computeBow(image, vocab, features[i]) # Rep is a list of descriptors for a given image
            train_rep.append(rep)
        BOW_train_features_path = SAVEPATH + 'bow_train_' + str(i) + '.npy'
        np.save(BOW_train_features_path, np.asarray(train_rep)) # Save the representations for vocabulary i
        train_rep = [] # reset the list to save the following vocabulary
        print("Saved BOW representations of all train images for vocabulary {} of {}.\t FILEPATH:{}".format(i+1, len(vocabularies), BOW_train_features_path))

        for image in test_images: # Compute the BOW representation of the testing set
            rep = computeBow(image, vocab, features[i])
            test_rep.append(rep)
        BOW_test_features_path = SAVEPATH + 'bow_test_' + str(i) + '.npy'
        np.save(BOW_test_features_path, np.asarray(test_rep)) # Save the representations for vocabulary i
        test_rep = [] # reset the list to save the following vocabulary
        print("Saved BOW representations of all test images for vocabulary {} of {}.\t FILEPATH:{}".format(i+1, len(vocabularies), BOW_test_features_path))
    print("BOW representations computed")
    '''

    ### -------------------------------- Part 3 -------------------------------------------###
    ### -----------------------------------------------------------------------------------###
    ### -----------------------------------------------------------------------------------###

    # Use BOW features to classify the images with a KNN classifier
    # A list to store the accuracies and one for runtimes
    knn_accuracies = []
    knn_runtimes = []

    # Your code below, eg:
    # for i, vocab in enumerate(vocabularies):
    # ... 
    # kmeans with test_rep, labels, and train_rep
    cur_test_rep = np.load(SAVEPATH + 'bow_test_' + '1' + '.npy')
    # print(len(cur_test_rep))
    import math 
    for val in cur_test_rep:
        if len(val) != 50:
            print(len(val))
        for elm in val:
            if math.isnan(elm):
                print("nan found")
                break
    
    for i in range(len(vocabularies)):
        for numNeighbors in [1, 3, 6]:
            start_time = time.time()
            cur_train_rep = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
            cur_test_rep = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')
            predictions = KNN_classifier(cur_train_rep, train_labels, cur_test_rep, numNeighbors)
            timeTaken = time.time() - start_time
            accuracy = reportAccuracy(test_labels, predictions)
            knn_accuracies.append(accuracy)
            knn_runtimes.append(timeTaken)
            print("KMEANS iteration {} for neighbors={} finished with accuracy={}, runtime={}".format(i+1, numNeighbors, accuracy, timeTaken))

    np.save(SAVEPATH+'knn_accuracies.npy', np.asarray(knn_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'knn_runtimes.npy', np.asarray(knn_runtimes)) # Save the runtimes in the Results/ directory
    


    ### -------------------------------- Part 4 -------------------------------------------###
    ### -----------------------------------------------------------------------------------###
    ### -----------------------------------------------------------------------------------###

    # Use BOW features to classify the images with 15 Linear SVM classifiers
    lin_accuracies = []
    lin_runtimes = []
    
    # Your code below
    #...

    np.save(SAVEPATH+'lin_accuracies.npy', np.asarray(lin_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'lin_runtimes.npy', np.asarray(lin_runtimes)) # Save the runtimes in the Results/ directory
    
    # Use BOW features to classify the images with 15 Kernel SVM classifiers
    rbf_accuracies = []
    rbf_runtimes = []
    
    # Your code below
    # ...
    
    np.save(SAVEPATH +'rbf_accuracies.npy', np.asarray(rbf_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH +'rbf_runtimes.npy', np.asarray(rbf_runtimes)) # Save the runtimes in the Results/ directory
            
    