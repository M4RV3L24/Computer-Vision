import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import ImageOps
import numpy as np
import cv2

def lib_apply_blur(image_array):
    # Apply Gaussian blur to the image
    return cv2.GaussianBlur(image_array, (5, 5), 0)

def lib_apply_sobel(image_array):
    # Apply Sobel filter to the image
    sobelX = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelX**2 + sobelY**2)
    return gradient_magnitude

def lib_applyfft(image_array, radius, filter_type='low'):
    # f = np.fft.fft2(image_array)
    # fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20*np.log(np.abs(fshift)+1)
    # return fshift, magnitude_spectrum
    # Perform FFT
    f = np.fft.fft2(image_array)
    fshift = np.fft.fftshift(f)
    
    # Get image dimensions
    rows, cols = image_array.shape
    crow, ccol = rows // 2 , cols // 2  # Center

    # Create a mask with the same dimensions as the image
    mask = np.zeros((rows, cols), np.uint8)

    # Apply the filter
    if filter_type == 'low':
        # Low-pass filter: Allow frequencies below the radius to pass
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    elif filter_type == 'high':
        # High-pass filter: Allow frequencies above the radius to pass
        mask = 1 - cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    else:
        raise ValueError("filter_type must be 'low' or 'high'")

    # Apply the mask to the shifted FFT
    fshift_filtered = fshift * mask

    # Calculate the magnitude spectrum for visualization
    magnitude_spectrum = 20 * np.log(np.abs(fshift_filtered))

    return fshift_filtered, magnitude_spectrum


def lib_reversefft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back



def lib_sharp(image_array):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image_array, -1, kernel)

def lib_customKernel(image_array, kernel_array):
    return cv2.filter2D(image_array, -1, kernel_array)

def lib_apply_threshold(image_array, threshold_value):
    return cv2.threshold(image_array, threshold_value, 255, cv2.THRESH_BINARY)[1]

def apply_threshold(image_array, threshold_value):
    # Apply a threshold to the image array
    return np.where(image_array >= threshold_value, 255, 0)

def apply_filter(image_array, kernel_array):
    image_height = image_array.shape[0]
    image_width = image_array.shape[1]

    kernel_height = kernel_array.shape[0]
    kernel_width = kernel_array.shape[1]

    offX = kernel_height//2
    offY = kernel_width//2

    processed_array = np.zeros((image_height, image_width), dtype=np.float32)
    # processed_array = np.copy(image_array)

    for row in range(image_height):
            for col in range(image_width):
                # Apply the kernel to the surrounding pixels
                sum = 0
                for rdif in range(-offX, offX+1):
                    for cdif in range(-offY, offY+1):
                        if (row + rdif) >= image_height or (row + rdif) < 0 or (col + cdif) >= image_width or (col + cdif) < 0:
                            continue
                        sum += image_array[row + rdif, col + cdif] * kernel_array[rdif + offX, cdif + offY]
                processed_array[row, col] = sum

    return processed_array
        

def apply_sobel_diagonal(image_array):
    sobel_kernelX = np.array(
        [[0, 1, 1], 
        [-1, 0, 1], 
        [-1, -1, 0]]
        )

    sobel_kernelY = np.array(
        [[1, 1, 1], 
        [0, 0, 0], 
        [-1, -1,- 1]]
        )

    # kernel = kernel/kernel.sum()
    

    # Get the dimensions of the image and kernel
    image_height = image_array.shape [0]
    image_width = image_array.shape [1]
    kernel_height, kernel_width = sobel_kernelX.shape

    # apply filter to image
    processed_array = np.zeros((image_height, image_width), dtype=np.float32)
    
    # Define the offset for the kernel (assuming the kernel is square and has odd dimensions)
    offX = kernel_height // 2
    offY = kernel_width // 2

    # # Iterate over each pixel in the image (excluding the borders)
    # # FOR RGB
    # for row in range(image_height):
    #     for col in range(image_width):
    #         # Apply the kernel to the surrounding pixels
    #         rgb = np.zeros(3)
    #         for rdif in range(-offX, offX+1):
    #             for cdif in range(-offY, offY+1):
    #                 if (row + rdif) >= image_height or (row + rdif) < 0 or (col + cdif) >= image_width or (col + cdif) < 0:
    #                     continue
    #                 for i in range(0, 3):
    #                     rgb[i] += image_array[row + rdif, col + cdif, i] * kernel[rdif + offX, cdif + offY]
            
    #         # Assign the result to the corresponding pixel in the output array
    #         processed_array[row, col] = rgb


    for row in range(image_height):
        for col in range(image_width):
            # Apply the kernel to the surrounding pixels
            sumX = 0
            sumY = 0
            for rdif in range(-offX, offX+1):
                for cdif in range(-offY, offY+1):
                    if (row + rdif) >= image_height or (row + rdif) < 0 or (col + cdif) >= image_width or (col + cdif) < 0:
                        continue
                    sumX += image_array[row + rdif, col + cdif] * sobel_kernelX[rdif + offX, cdif + offY]
                    sumY += image_array[row + rdif, col + cdif] * sobel_kernelY[rdif + offX, cdif + offY]

            # Assign the result to the corresponding pixel in the output array
            gradient_magnitude = np.sqrt(sumX**2 + sumY**2)
            processed_array[row, col] = gradient_magnitude
    
    
    return processed_array


def apply_sharp(image_array):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return apply_filter(image_array, kernel)


def apply_gaussian(image_array):
    kernel = np.array([[1/16, 2/16, 1/16],
                       [2/16, 4/16, 2/16],
                       [1/16, 2/16, 1/16]])
    return apply_filter(image_array, kernel)