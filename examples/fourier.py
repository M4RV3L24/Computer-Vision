import math
import cv2 as cv
import numpy as np


def convolute(img, kernel, strength=1):
    b, g, r = cv.split(img)
    
    # Apply the kernel to each channel separately
    b_filtered = cv.filter2D(b, -1, kernel)
    g_filtered = cv.filter2D(g, -1, kernel)
    r_filtered = cv.filter2D(r, -1, kernel)

    # Merge the channels back
    img2 = cv.merge([b_filtered, g_filtered, r_filtered])
    return img2

def convolute_gray(img, kernel):
    # Apply the kernel to the grayscale image
    filtered_img = cv.filter2D(img, -1, kernel)
    
    return filtered_img

def fft(img):
    f = np.fft.fft2(img)            # Compute the 2D FFT
    fshift = np.fft.fftshift(f)     # Shift zero-frequency components to the center
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    magnitude= magnitude.astype(np.uint8)
    return fshift , magnitude

def reverse_fft(fshift):
    f_ishift = np.fft.ifftshift(fshift)  # Inverse shift
    img_back = np.fft.ifft2(f_ishift)    # Compute the 2D IFFT
    img_back = np.real(img_back)         # Take the real part of the inverse transform
    return img_back

x_mouse, y_mouse = (None, None)

img = cv.imread(cv.samples.findFile("examples/images/cat.jpg"))
img =  cv.cvtColor(img, cv.COLOR_BGR2GRAY)

fshift, magnitude = fft(img)

img2 = img
img2 = reverse_fft(fshift)
img2 = cv.normalize(img2, None, 0, 255, cv.NORM_MINMAX)
img2 = np.uint8(img2)

img3 = magnitude

img4 = np.zeros_like(img)

combined_image = np.vstack((np.hstack((img, img2)), np.hstack((img3, img4))))

cv.namedWindow(winname='combined', flags=cv.WINDOW_NORMAL)

height, width = combined_image.shape[:2]

resize = 1
cv.resizeWindow(winname='combined', width=int(width*resize), height=int(height*resize))

while True:
    cv.imshow('combined', combined_image)
    k = cv.waitKey(10) & 0xFF
    if k == ord('q'):
        break

cv.destroyAllWindows()


