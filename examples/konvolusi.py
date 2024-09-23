import cv2
import numpy as np

# Load your image
image_path = 'examples/images/applelogo.jpg'  # Replace with your image path
img = cv2.imread(image_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_copy = img.copy()

# convolution kernel
convolution_kernel = np.array([[1/35, 1/35, 1/35, 1/35, 1/35, 1/35, 1/35], 
                               [1/35, 1/35, 1/35, 1/35, 1/35, 1/35, 1/35], 
                               [1/35, 1/35, 1/35, 1/35, 1/35, 1/35, 1/35],
                               [1/35, 1/35, 1/35, 1/35, 1/35, 1/35, 1/35],
                               [1/35, 1/35, 1/35, 1/35, 1/35, 1/35, 1/35]])
convolution_kernel2 = np.array([[0,-1/8,0],
                                [-1/8,1.5,-1/8],
                                [0,-1/8,0]])
k = int((len(convolution_kernel)-1)/2)
l = int((len(convolution_kernel[0])-1)/2)
k2 = int((len(convolution_kernel2)-1)/2)
l2 = int((len(convolution_kernel2[0])-1)/2)

# add kernel to the image
for i in range(k, img.shape[0]-k):
    for j in range(l, int(img.shape[1]/2)-l):
        sum = 0
        for x in range(-k, k+1):
            for y in range(-l, l+1):
                sum += img[i+x, j+y] * convolution_kernel[x+k, y+l]
        img_copy[i, j] = sum

for i in range(k2, img.shape[0]-k2):
    for j in range(l2 + int(img.shape[1]/2), img.shape[1]-l2):
        sum = 0
        for x in range(-k2, k2+1):
            for y in range(-l2, l2+1):
                sum += img[i+x, j+y] * convolution_kernel2[x+k2, y+l2]
        img_copy[i, j] = sum

# Display the image
combined_img = np.hstack((img, img_copy))

scale_percent = 20  # Adjust this percentage to resize the image as needed
width = int(combined_img.shape[1] * scale_percent / 100)
height = int(combined_img.shape[0] * scale_percent / 100)
dim = (width, height)
img_resized = cv2.resize(combined_img, dim, interpolation=cv2.INTER_AREA)

# Create a window
cv2.namedWindow('Image')

# Display the combined image
cv2.imshow('Image', img_resized)
cv2.waitKey(0)  # Wait for a key press to exit
cv2.destroyAllWindows()
