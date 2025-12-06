# Copied from https://github.com/calvinbakker/structuralComplexity
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cmasher as cms

def importImage(path,option):
    r'''
    This function is a template for importing an image such
    that the structural complexity can be calculated.
    
    '''
    # import image from path
    image = mpimg.imread(path)

    # convert to size (N,N)
    if   option=='red':
        array = image[:,:,0]
    elif option=='blue':
        array = image[:,:,1]
    elif option=='green':
        array = image[:,:,2]
    elif option=='intensity':
        array = np.mean(image,axis=2)

    # normalize between [-1,1]
    nArray = array/128 - 1

    return nArray

def catchErrors(image,kLargerThan,kMax):
        r'''
        This function catches errors in the structuralComplexity()
        function inputs.

        '''
        imageSize = image.shape[0]
        kLargerThan = int(kLargerThan)
        kMax = int(kMax)

        if (imageSize & (imageSize - 1)) != 0:
            raise Exception("🤖 Bleep bloop.. ERROR image-size is not a power of 2") 

        if kLargerThan < 0:
            raise Exception("🤖 Bleep bloop.. ERROR kLargerThan value is too small")  
        
        if kMax > int(np.log2(imageSize))-2:
            raise Exception("🤖 Bleep bloop.. ERROR kMax value is too large") 

        if kMax <= kLargerThan:
            raise Exception("🤖 Bleep bloop.. ERROR k-range not ok") 
        return None

def structuralComplexity(image,kLargerThan,kMax):
    r'''
    This function calculates the structural complexity as defined
    in the paper: Bagrov, Andrey A., et al. "Multiscale structural 
    complexity of natural patterns" (10.1073/pnas.2004976117).

    The input of the function should be a Numpy array with size 
    (N,N), where N is a power of 2. The values in the array 
    should be normalized such that all values are floats in the
    range of [-1,1].

    The function returns the structural complexity $C$, where 
    $C_\lambda$ is computed by the function C_lambda(), and the 
    summation is performed as given by equation [4] in the paper.

    '''

    # local functions
    def coarseGrainBlock(i):
        r'''
        This function performs the coarse-graining of an image.

        Parameters:
        · N is the size of the image.
        · n is the size of the coarse-graining.

        The steps:
        · Reshape the (N,N) image into (n,n) blocks 
        · For each (n,n) block, reshape from (n,n) into (n*n)
        · Take np.mean() of each block
        · Reshape back into a square of size (N/2,N/2)
        · Apply the Kronecker product with a matrix of 1's of
          size (n,n), to get back the original size of (N,N)
        
        '''
        assert image.shape[0] == image.shape[1] # 只考虑方阵
        N = image.shape[0]
        n = 1<<i 
        reshape_image = image.reshape(N//n, n, N//n, n)
        reshape_image = reshape_image.swapaxes(1,2)
        #reshape_image = reshape_image.reshape(-1, n*n)
        #reshape_image = np.mean(reshape_image, axis=1)
        #reshape_image = reshape_image.reshape(N//n,N//n)
        # 这一行可以替代前三行的内容
        reshape_image = np.mean(reshape_image, axis=(2,3))
        coarseGrainMat = np.kron(reshape_image, np.ones((n,n)))  
        return coarseGrainMat

    def O(k1,k2):
        r''' 
        This function implements equation [3] of the paper.
        The correspondence in hidden in the coarseGrainBlock()
        function.

        '''
        
        return coarseGrainBlock(k1)*coarseGrainBlock(k2)

    def C_lambda(k):
        r'''
        This function returns the value of $C_\lambda$. It 
        implements equation [4] of the paper.

        · np.mean implements the summation over $k$
        · O(i,j) implements the the object $O_{i,j}$ of 
          equation [3] in the paper.

        '''
        return np.mean(
            np.abs(O(k+1,k) - (O(k,k)+O(k+1,k+1))/2)
            )

    # catch errors in the function inputs
    catchErrors(image,kLargerThan,kMax)
    # compute the structural complexity
    C = 0
    for k in range(kLargerThan,kMax):
        c_img = coarseGrainBlock(k)
        # plt.imshow(c_img, cmap=cms.prinsenvlag)
        # plt.axis("Off")
        # plt.title("imageArray")
        # plt.colorbar(shrink=0.5)
        # plt.clim(-1,1)
        # plt.show()
        C+=C_lambda(k)
    return C

if __name__ == "__main__":
    
    # Template for importing a .jpeg image
    path = 'images/corsica.jpeg' 
    imageArray = importImage(path, 'intensity')

    # Plot the imageArray
    plt.figure(figsize=[5,5], dpi=100)
    plt.imshow(imageArray, cmap=cms.prinsenvlag)
    plt.axis("Off")
    plt.title("imageArray")
    plt.colorbar(shrink=0.5)
    plt.clim(-1,1)
    plt.show()

    # Set the range of value $k$ in equation [4] in the paper
    kLargerThan = 0
    kMax = int(np.log2(imageArray.shape[0])) - 2  # For the full range maximal value
    
    C = structuralComplexity(imageArray, kLargerThan=0, kMax=kMax)

    print(f"Structural complexity of the image is: {C.round(3)}")