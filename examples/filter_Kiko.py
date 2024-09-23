import cv2
import numpy as np

def convolution(kernel, arrImg):
    kernelHeight, kernelWidth = kernel.shape
    imageHeight, imageWidth = arrImg.shape

    # Calculate padding size
    padHeight = kernelHeight // 2
    padWidth = kernelWidth // 2

    # Pad the input image
    paddedImg = np.pad(arrImg, ((padHeight, padHeight), (padWidth, padWidth)), mode='constant', constant_values=0)

    output = np.zeros((imageHeight, imageWidth), dtype=np.float32)

    for i in range(imageHeight):
        for ii in range(imageWidth):
            yStart = i
            yEnd = i + kernelHeight
            xStart = ii
            xEnd = ii + kernelWidth

            roi = paddedImg[yStart:yEnd, xStart:xEnd]
            output[i, ii] = np.sum(kernel * roi)

    # Normalize the output to the range [0, 255]
    output = np.clip(output, 0, 255)

    # Convert the result to uint8
    output = output.astype(np.uint8)

    return output

def generateIdentitiyKernel():
    kernel = np.array([
        [0,0,0],
        [0,1,0],
        [0,0,0]
    ])
    return kernel

def generateGaussianBlurKernel(n = 3, sigma = 1):
    ax = np.linspace(-(n - 1) / 2., (n - 1) / 2., n)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    
    kernel = kernel / np.sum(kernel)
    
    return kernel

def generateAverageBlurKernel(n = 3):
    kernel = np.ones((n,n))
    
    kernel = kernel / np.sum(kernel)
    
    return kernel

def generateSharpenKernel():
    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    return kernel

def generateEdgeDetectionKernel():
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    return kernel

kernel1 = generateEdgeDetectionKernel()

img = cv2.imread('examples/images/cat.jpg', cv2.IMREAD_GRAYSCALE)

img_result = convolution(kernel1, img)

cv2.imshow('before', img)
cv2.imshow('convolution', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
