import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def on_hover(event):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        # Get the pixel intensity (ensure indices are within image bounds)
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            intensity = image[y, x]
            normalized_intensity = intensity  # Normalize to [0, 1]
            # Display the information
            print(f"Position: ({x}, {y}), Intensity: {normalized_intensity:.2f}")
            ax.set_title(f"Position: ({x}, {y}), Intensity: {normalized_intensity:.2f}")
            fig.canvas.draw_idle()

# Load the image
# Replace 'path_to_image.png' with the actual path to your grayscale image
image_path = 'images/dog-grayscale.jpeg'
image = mpimg.imread(image_path)

# Check if image is already grayscale
if len(image.shape) > 2 and image.shape[2] == 3:
    # Convert to grayscale using luminosity method
    image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

# Set up the plot
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray', vmin=0, vmax=255)
ax.axis('off')

# Connect the hover event
fig.canvas.mpl_connect('motion_notify_event', on_hover)

plt.show()
