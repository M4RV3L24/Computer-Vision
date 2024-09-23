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

def blur(img, ksize=5):
    """
    Apply an average blur to the image using a custom kernel.
    
    Parameters:
    img (numpy.ndarray): The input image.
    ksize (int): The size of the kernel. Must be an odd number. Default is 5.
    
    Returns:
    numpy.ndarray: The blurred image.
    """
    if ksize % 2 == 0:
        ksize += 1  # Ensure the kernel size is odd

    # Define an averaging kernel
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    
    img2 = convolute(img, kernel, ksize)
    return img2


def sharpen(img, strength=1):
    # Ensure strength is within a reasonable range
    strength = max(1, min(strength, 10))  # Clamping the value between 1 and 10
    
    # Dynamic adjustment of alpha and beta
    alpha = 1.0 + (strength * 0.2)  # Increase original image weight with higher strength
    beta = strength * 0.2  # Increase sharpened image weight with higher strength

    # Define a custom-shaped kernel
    # Ensuring kernel is adjusted dynamically based on strength
    kernel = np.array([
        [0, -1 * strength, 0],
        [-1 * strength, 4 + 4 * strength, -1 * strength],
        [0, -1 * strength, 0]
    ], dtype=np.float32)
    
    img2 = convolute(img, kernel)
    
    # Blend the original and sharpened images
    img_sharpened = cv.addWeighted(img, alpha, img2, beta, 0)
    
    return img_sharpened


def generate_gaussian_kernel(ksize, sigma=1.0):
    """
    Generate a Gaussian kernel.
    
    Parameters:
    ksize (int): The size of the kernel. Must be an odd number.
    sigma (float): The standard deviation of the Gaussian distribution.
    
    Returns:
    numpy.ndarray: The Gaussian kernel.
    """
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussian_blur(img, ksize=5, sigma=1.0):
    """
    Apply Gaussian blur to the image using a custom Gaussian kernel.
    
    Parameters:
    img (numpy.ndarray): The input image.
    ksize (int): The size of the kernel. Must be an odd number. Default is 5.
    sigma (float): The standard deviation for Gaussian kernel. Default is 1.0.
    
    Returns:
    numpy.ndarray: The blurred image.
    """
    if ksize % 2 == 0:
        ksize += 1  # Ensure the kernel size is odd

    # Generate Gaussian kernel
    kernel = generate_gaussian_kernel(ksize, sigma)
    
    img2 = convolute(img, kernel, ksize)
    return img2

def generate_motion_blur_kernel(length, angle=0):
    """
    Generate a motion blur kernel.
    
    Parameters:
    length (int): The length of the blur effect.
    angle (int): The angle of the blur direction in degrees.
    
    Returns:
    numpy.ndarray: The motion blur kernel.
    """
    if length % 2 == 0:
        length += 1  # Ensure the kernel size is odd for symmetry

    kernel = np.zeros((length, length), dtype=np.float32)

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Calculate center of the kernel
    center = length // 2

    # Generate the kernel
    for i in range(length):
        for j in range(length):
            # Check if the point is along the direction of the line
            if np.abs((i - center) * np.sin(angle_rad) - (j - center) * np.cos(angle_rad)) < 1:
                kernel[i, j] = 1

    # Normalize the kernel
    kernel /= np.sum(kernel)
    
    return kernel

def motion_blur(img, length=7, angle=0):
    """
    Apply motion blur to the image using a custom kernel.
    
    Parameters:
    img (numpy.ndarray): The input image.
    length (int): The length of the blur effect.
    angle (int): The angle of the blur direction in degrees.
    
    Returns:
    numpy.ndarray: The motion blurred image.
    """
    # Generate the motion blur kernel
    kernel = generate_motion_blur_kernel(length, angle)
    
    # Apply convolution with the motion blur kernel
    img_blurred = convolute(img, kernel)
    
    return img_blurred

def pixelate(img, pixel_size=10):
    """
    Apply pixelation to the image.
    
    Parameters:
    img (numpy.ndarray): The input image.
    pixel_size (int): The size of the pixelation blocks.
    
    Returns:
    numpy.ndarray: The pixelated image.
    """
    height, width, _ = img.shape
    # Resize the image to a smaller size
    small_img = cv.resize(img, (width // pixel_size, height // pixel_size), interpolation=cv.INTER_LINEAR)
    # Resize back to original size
    pixelated_img = cv.resize(small_img, (width, height), interpolation=cv.INTER_NEAREST)
    return pixelated_img

# robert, prewitt, sobel canny

def gradient(img, kernel_x, kernel_y):

    # Convert the image to grayscale if it's not already
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img = img

    # Compute the gradients in the x and y directions using the convolute function
    grad_x = convolute_gray(img, kernel_x)
    grad_y = convolute_gray(img, kernel_y)

    # Compute the gradient magnitude
    gradient_magnitude = cv.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))

    # Normalize the result to fit within the range of 0 to 255
    gradient_magnitude = cv.normalize(gradient_magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Convert to uint8
    gradient_magnitude = np.uint8(gradient_magnitude)

    # Return the gradient magnitude as an image
    return gradient_magnitude

def robert(img):
    # Define Robert's Cross kernels
    roberts_cross_x = np.array([[1, 0], [0, -1]])
    roberts_cross_y = np.array([[0, 1], [-1, 0]])
    return gradient(img, roberts_cross_x, roberts_cross_y)

def prewitt(img):
    # Define Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    return gradient(img, prewitt_x, prewitt_y)

def sobel(img):
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return gradient(img, sobel_x, sobel_y)

def canny(img):
    
    if len(img.shape) == 3:
        # Split the image into its B, G, R channels
        b, g, r = cv.split(img)

        # Apply Canny edge detection to each channel
        edges_b = cv.Canny(b, 100, 200)
        edges_g = cv.Canny(g, 100, 200)
        edges_r = cv.Canny(r, 100, 200)

        # Merge the channels back together
        edges = cv.merge([edges_b, edges_g, edges_r])

        return edges
    
    # Apply Gaussian blur to reduce noise
    blurred_img = cv.GaussianBlur(img, (5, 5), 1.4)

    # Use Canny edge detector
    return cv.Canny(blurred_img, 100, 200)


def create_rotated_kernel(angle_degrees):
    """
    Create a rotated edge detection kernel based on the angle.

    Parameters:
    angle_degrees (float): The angle of the edge detection kernel in degrees.

    Returns:
    numpy.ndarray: The rotated kernel.
    """
    # Define a base kernel for edge detection (e.g., Sobel operator)
    base_kernel = np.array([
        [0, -1, 0],
        [1, 0, -1],
        [0, 1, 0]
    ], dtype=np.float32)

    # Create a rotation matrix
    # angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = cv.getRotationMatrix2D((1, 1), angle_degrees, 1)

    # Rotate the kernel
    rotated_kernel = cv.warpAffine(base_kernel, rotation_matrix, (3, 3), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    
    return rotated_kernel

def custom_edge_detection(img, angle=0):
    """
    Apply custom edge detection based on a rotated kernel.

    Parameters:
    img (numpy.ndarray): Input image (grayscale or color).
    angle (float): Angle for edge detection in degrees.

    Returns:
    numpy.ndarray: The resulting image after applying the custom edge detection.
    """
    # Convert to grayscale if the input image is colored
    if len(img.shape) == 3:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Apply Gaussian blur to reduce noise
    blurred_img = cv.GaussianBlur(gray_img, (5, 5), 1.4)

    # Create the rotated kernel
    kernel = create_rotated_kernel(angle)

    # Apply the kernel to the grayscale image
    filtered_img = cv.filter2D(blurred_img, -1, kernel)

    # Normalize the result to fit within the range of 0 to 255
    filtered_img = cv.normalize(filtered_img, None, 0, 255, cv.NORM_MINMAX)

    # Convert to uint8
    filtered_img = np.uint8(filtered_img)

    return filtered_img



x_mouse, y_mouse = (None, None)

img = cv.imread(cv.samples.findFile("examples/images/cat.jpg"))
img =  cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Split the image into its B, G, R channels

img2 = custom_edge_detection(img, 180)
# img3 = prewitt(img)
# img4 = custom_edge_detection(img, 0)

# combined_image = np.vstack((np.hstack((img, img2)), np.hstack((img3, img4))))
combined_image = np.hstack((img, img2))

cv.namedWindow(winname='combined', flags=cv.WINDOW_NORMAL)

height, width = combined_image.shape[:2]

resize = 0.8
cv.resizeWindow(winname='combined', width=int(width*resize), height=int(height*resize))

i = 0
while True:
    cv.imshow('combined', combined_image)
    i+=4
    # print(i % 360)
    img2 = custom_edge_detection(img, i % 360)
    combined_image = np.hstack((img, img2))
    
    k = cv.waitKey(10) & 0xFF
    if k == ord('q'):
        break

cv.destroyAllWindows()


