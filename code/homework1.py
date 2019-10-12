from classifiers import *

# interpreting your performance with 100 training examples per category:
# accuracy  =   0 ->  your code is broken (probably not the classifier's
#                     fault! a classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .07 -> your performance is chance.
#  accuracy ~= .20 -> rough performance with tiny images and nearest
#                     neighbor classifier.
#  accuracy ~= .20 -> rough performance with tiny images and linear svm
#                     classifier. the linear classifiers will have a lot of
#                     trouble trying to separate the classes and may be
#                     unstable (e.g. everything classified to one category)
#  accuracy ~= .50 -> rough performance with bag of sift and nearest
#                     neighbor classifier.
#  accuracy ~= .60 -> you've gotten things roughly correct with bag of
#                     sift and a linear svm classifier.
#  accuracy >= .70 -> you've also tuned your parameters well. e.g. number
#                     of clusters, svm regularization, number of patches
#                     sampled when building vocabulary, size and step for
#                     dense sift features.
#  accuracy >= .80 -> you've added in spatial information somehow or you've
#                     added additional, complementary image features. this
#                     represents state of the art in lazebnik et al 2006.
#  accuracy >= .85 -> you've done extremely well. this is the state of the
#                     art in the 2010 sun database paper from fusing many 
#                     features. don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> you get to teach the class next year.
#  accuracy >= .96 -> you can beat a human at this task. this isn't a
#                     realistic number. some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.


if __name__ == "__main__":
	img = cv2.imread('data/train/bedroom/image_0001.jpg', cv2.IMREAD_UNCHANGED)
	resized = imresize(img, 100)
	cv2.imwrite('output.jpg', resized)

	# resize/normalize training images and put into separate list
	# creates corresponding list of the same size that holds integer value of categories
	import os
	rootdir = 'data/train'

	categories = ['Forest', 'bedroom', 'Office', 'Highway', 'Coast', 'Insidecity', 'TallBuilding', 'industrial', 'Street', 'livingroom', 'Suburb', 'Mountain', 'kitchen', 'OpenCountry', 'store']

	processedImages = []
	processedImageLabels = []

	currentCategoryID = 0
	for subdir, dirs, files in os.walk(rootdir):
		# print("Debug | current directory is:" + subdir + " and currentCategoryID is:" + str(currentCategoryID))
		for file in files:
			imageFilePath = os.path.join(subdir, file)
			if imageFilePath.lower().endswith(('.png', '.jpg', '.jpeg')):
					currentImage = cv2.imread(imageFilePath, cv2.IMREAD_UNCHANGED)
					currentResizedImage = imresize(currentImage, 100)
					processedImages.append(currentResizedImage)
					processedImageLabels.append(currentCategoryID)
				#print(imageFilePath)
		currentCategoryID = currentCategoryID + 1

	# print("len(processedImages): " + str(len(processedImages)))
	# print("len(processedImageLabels): " + str(len(processedImageLabels)))