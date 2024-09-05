from PIL import Image
import numpy as np

def readImage(path):
    image = Image.open(path)
    image_array = np.array(image)
    rows, cols, channels = image_array.shape
    return image_array, rows, cols

def saveImage(array, path, show=False):
    image = Image.fromarray(array)
    image.save(path)
    if show:
        image.show()
        
def combineImage(path1, path2, combinedPath, show=False):
    image1 = Image.open(path1)
    image2 = Image.open(path2)

    image1 = image1.resize((image1.width, image2.height)) if image1.height != image2.height else image1
    image2 = image2.resize((image2.width, image1.height)) if image2.height != image1.height else image2

    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)

    new_image = Image.new("RGB", (total_width, max_height))

    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))

    new_image.save(combinedPath)
    if show:
        new_image.show()