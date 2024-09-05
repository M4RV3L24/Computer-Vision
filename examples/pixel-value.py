from PIL import Image
import numpy as np

# Load the image
image_path = 'images/dog-grayscale.jpeg'  # Replace with your image path
image = Image.open(image_path)

# Ensure the image is grayscale
image = image.convert('L')

# Convert image to numpy array
image_array = np.array(image)

# Get the dimensions of the image
height, width = image_array.shape

# Print each pixel value
# for y in range(height):
#     for x in range(width):
#         print(f'Pixel at ({x}, {y}): {image_array[y, x]}'
        
# for inverted image
inverted_array = np.array(image_array)
for y in range(height):
    for x in range(width//2):
        inverted_array[y, x] = 255 - inverted_array[y, x]
        
# for y in range(height):
#     for x in range(1):
#         print(f'Pixel at ({x}, {y}): {image_array[y, x]}')
#         print(f'Pixel at ({x}, {y}): {inverted_array[y, x]}')

inverted_image = Image.fromarray(inverted_array, 'L')
inverted_image.save('images/inverted-dog.png')
# image.show()
# inverted_image.show()



# Combining 2 images in 1 image
# Load the two grayscale images
image1 = Image.open('images/dog-grayscale.jpeg').convert('L')  # Ensure it's grayscale
image2 = Image.open('images/inverted-dog.png').convert('L')  # Ensure it's grayscale

# Ensure both images are of the same height (resize if necessary)
image1 = image1.resize((image1.width, image2.height)) if image1.height != image2.height else image1
image2 = image2.resize((image2.width, image1.height)) if image2.height != image1.height else image2

# Create a new image with a width that's the sum of both images and the same height
total_width = image1.width + image2.width
max_height = max(image1.height, image2.height)

# Create a blank image in grayscale mode
new_image = Image.new('L', (total_width, max_height))

# Paste the two images into the new image
new_image.paste(image1, (0, 0))
new_image.paste(image2, (image1.width, 0))

# Save or display the combined image
new_image.save('images/combined-half.jpg')
new_image.show()

print(width)

