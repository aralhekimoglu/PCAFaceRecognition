import numpy as np
import matplotlib.image

def readImages():
    images=[]
    classes=[]
    flattenedImagesOfClasses=[]
    for i in range (1,41):
        classes.append(i)
        mat=np.zeros((9,10304))
        for j in range (1,10):
            source='dataset/training/s'+str(i)+'/'+str(j)+'.pgm'
            read_img = matplotlib.image.imread(source)
            images.append(read_img)
            mat[j-1]=read_img.flatten()
        flattenedImagesOfClasses.append(mat)
    
    return images,classes,flattenedImagesOfClasses

def argsortTwoArrays(a,b,ascendingOrder=False):
    """
        Sorts array b in argument order to sort array a
        Args:
            a(np.array): Array to use the argsort order to sort the other array
            b(np.array): Array to be sorted using the argsort order of other array
            ascendingOrder(boolean): Determine whether a should be sorted in descending/ascending order
        Returns:
            b_sorted(np.array): Sorted array of b
    """
    if ascendingOrder==True:
        a=np.array(a)[::-1]
    else:
        a=np.array(a)
    a=np.array(a)
    b=np.array(b)
    sortingOrder=np.argsort(a)
    b_sorted=b[sortingOrder]
    
    return b_sorted