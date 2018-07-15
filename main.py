import numpy as np
from Utils import readImages,argsortTwoArrays
import matplotlib.image

class EigenfaceClassifier():
    
    def train(self,k=400):
        """
            Apply pca to training images,obtain top k eigenfaces and use them to reduce dimensions of images.
        Args:
            k(int): number of eigenfaces to use in training
        Returns: 
            an instance of self
        """  
        images,self.classes,flattenedImagesOfClasses = readImages()   

        imagesMatrix=np.array([np.array(images[i]).flatten() for i in range (0,len(images)) ])
       
        #PCA
        U, D, V = np.linalg.svd(imagesMatrix)
        
        self.eigenfaces = V[0:k]         
        
        self.classProjected=[self.project(flattenedImagesOfClass).mean(0) for flattenedImagesOfClass in flattenedImagesOfClasses] 
        
        return self
        
    def project(self, X):
        """
            Project the given image/series of images X to a lower dimension 
            spanned by top eigenfaces defined in pca.  
        Args:
            X(np.array) of shape (number of similar images in the class,resolution of the image) 
        Returns: 
            (np.array) of shape (number of similar images in the class,number of eigenfaces)  
        """        
        return (X).dot(self.eigenfaces.T)

    def predict(self, imageSource):
        """
            Predict the class of the image by calculating the distance of the projected target image 
            to projected mean of each class and determine the minimum distance. 
            
            Args:
                imageSource (str): address of the target Image
            Returns:
                predictedClass (int): Returns the class most similar to target image
        """
        targetImage = matplotlib.image.imread(imageSource).flatten()
        targetProjectedImage = self.project(targetImage)
        distances=[]
        classesArray=[]
        
        for i in range(len(self.classProjected)):
            distance = np.linalg.norm(targetProjectedImage - self.classProjected[i])
            distances.append(distance)
            classesArray.append(self.classes[i])
                            
        predictedClass=argsortTwoArrays(distances,classesArray)[0]
        return predictedClass

#Train classifierd
classifier = EigenfaceClassifier()
classifier.train(k=400)

#Run trained classifier on test sets
errorNum=0

for i in range (1,40):
    source='dataset/test/s'+str(i)+'/10.pgm'
    prediction = classifier.predict(source)
    if i !=prediction:
        errorNum+=1

print "Accuracy for test set is %",(100-errorNum/40.0*100)
