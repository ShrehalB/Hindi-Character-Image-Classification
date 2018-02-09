# Hindi-Character-Image-Classification
Machine Learning Project

1. Data Generation : Mobile phone captured images of old manuscript written in Devnagri is taken. Page by page , Line by
	 Line word by word every character / alphabet of Devnagri script is cropped. A total of 1966 character image Generation.

2. Labeling of images - every extracted character is manually labelled as per the character code table of 'The Unicode 
	Standard' Version 10.0. This labeling is attached in the file name of the image (of the format page0_0_1_2361_2380 
	where 2361 and 2380 are the labels of two characters noticed in the image )

3. Pre-process : for every image OpenCV Library of Python.

	a. Noise is removed
	b. outliers are filtered 
	c. dimensions are fixed with padding
	d. binarization of multicolored images -  colors used, 1 for background (absolute white), one for main image 
		(absolute black) 
	e. Image transformation : every single image is used to create multiple images of same character by slight rotation, translation     shifting etc. inorder to obtain a significant magnitude of training data. 


4. Storage : pre-processed images and corresponding labels are stored in the form of Arrays. 2 -dimensional images represented in the form of 0 and 1 form an element of the Image array. An 1-D array of len (= total classes possible) forms an element of the label array

5. Model : A Deep learning model is trained on the available training data. Its hyperparameters are tuned with repeated experimentations on Cross validation set. Input of the model is array of images, predicted output is the array mentioning the probablity of occurence of each class in that Image.
 	A threshold of 0.5 is maintained for an image to have a perticular class.


6.  Evaluation : criteria used is Accuracy measure.
