import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# Detect keypoints using FAST
def detect_fast_keypoints(image):
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(image, None)
    return keypoints

# Detect keypoints using ORB
def detect_orb_keypoints(image):
    orb = cv2.ORB_create()  
    keypoints = orb.detect(image, None)
    return keypoints

# Detect keypoints using SIFT (for orientation information)
def detect_sift_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    return keypoints

# Compute BRIEF descriptors
def compute_brief_descriptors(image):
    # Initialize BRIEF descriptor
    # star = cv2.xfeatures2d.StarDetector_create() 
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # Compute descriptors and keypoint
    kp = detect_orb_keypoints(image)
    # kp, descriptors = brief.compute(image, kp)

    # Initialize ORB to detect keypoint orientations
    orb = cv2.ORB_create()

    # Compute the orientation of the keypoints
    keypoints, _ = orb.compute(image, kp)
    
    # Compute the descriptors
    descriptors_list = []
    for kp in keypoints:
        # Extract the keypoint orientation
        angle = kp.angle

        # Compute the rotation matrix for the keypoint
        M = cv2.getRotationMatrix2D((kp.pt[0], kp.pt[1]), angle, 1)

        # Apply the rotation matrix to the keypoint's patch
        patch = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Compute the BRIEF descriptor on the rotated patch
        kp_rotated, desc = brief.compute(patch, [kp])
        
        if desc is not None:
            descriptors_list.append(desc)

    descriptors = np.array(descriptors_list).reshape(-1, 32)  # Convert list to array

    return keypoints, descriptors

        



    # keypoints_without_size = np.copy(image)
    # # keypoints_with_size = np.copy(image)
    # cv2.drawKeypoints(image, kp, keypoints_without_size, color = (0, 255, 0))
    # return kp, descriptors


# Compute FREAK descriptors
def compute_freak_descriptors(image, keypoints):
  freak = cv2.xfeatures2d.FREAK_create()
  keypoints, descriptors = freak.compute(image, keypoints)
  return keypoints, descriptors

# Match descriptors using BFMatcher
def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Match image using BRIEF descriptors
def brief_feature_matching(image1, image2):
    kp1, desc1 = compute_brief_descriptors(image1)
    kp2, desc2 = compute_brief_descriptors(image2)

    matches = match_descriptors(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_img = cv2.drawMatches(image1, kp1, image2, kp2, matches[:100], None, flags=2)
    return match_img, kp1, kp2

# Match image using ORB descriptors
def orb_feature_matching(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    matches = match_descriptors(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_img = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=2)
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

    # Create test image by adding Scale Invariance and Rotational Invariance
    # rotated_image = cv2.pyrDown(image)
    # rotated_image = cv2.pyrDown(rotated_image)
    num_rows, num_cols = image.shape[:2]

    # Rotate the image
    M = cv2.getRotationMatrix2D((num_cols//2, num_rows//2), rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (num_cols, num_rows))

    # Match features
    # match_img, kp1, kp2 = orb_feature_matching(image, rotated_image)
    match_img, kp1, kp2 = brief_feature_matching(image, rotated_image)

    

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
