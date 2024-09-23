import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Function to perform ORB feature matching
def orb_feature_matching(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_img = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=2)
    return match_img, kp1, kp2

# Load the image
image = cv2.imread('./examples/images/twitterlogo.jpg')

# Create a figure for displaying the images
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

# Initial rotation angle
rotation_angle = 0

# Function to update the display when the slider is adjusted
def update(val):
    global rotation_angle
    rotation_angle = slider.val

    # Rotate the image
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Match features
    match_img, kp1, kp2 = orb_feature_matching(image, rotated_image)

    # Update the display
    ax.clear()
    ax.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    ax.set_title('Feature Matching between Original and Rotated Images')
    ax.axis('off')
    plt.draw()

# Create a slider for rotation angle
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
slider = Slider(ax_slider, 'Rotation Angle', -180, 180, valinit=0)
slider.on_changed(update)

# Show the initial match
update(0)

plt.show()
