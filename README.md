# PRML-Algorithms

My implementation of the eigenfaces algorithm used in face recognition. <br />

Model is trained using PCA to reduce dimensions of 92x112 images to smaller dimensions. At training step eigenfaces(eigenvectors resulting from PCA) are learned.

To make predictions, model projects the target image using eigenfaces learned in training and calculates the distance in the smaller dimensional space. Closest distance is predicted to be the most probable class.

This model achieves a test set error of 92.5%. 

## Required Packages

numpy <br />
matplotlib <br />
scikit-learn

## Dataset

The ORL Database of Faces - http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html <br />

Each person has 10 images and training set is the first 9 images, test set is the last image.