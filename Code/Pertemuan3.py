import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

def apply_gaussian_blur(image_array, sigma=1):
    # Apply Gaussian blur to the image
    return gaussian_filter(image_array, sigma=sigma)


def load_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    
    if file_path:
        # Load and resize the selected image
        global image, tk_before, tk_after
        image = Image.open(file_path)

        # Convert the image to grayscale
        image = image.convert("L")

        # Add a border to the image
        border_width = 10  # Adjust the border width as needed
        border_color = "black"  # Adjust the border color as needed
        image = ImageOps.expand(image, border=border_width, fill=border_color)
        
        # Resize the image to fit the canvas while maintaining aspect ratio
        image.thumbnail((canvas_width, canvas_height))

        # Convert the image to a NumPy array
        image_array = apply_gaussian_blur(np.array(image))

        # Apply Sobel filter using the library function
        sobel_x = sobel(image_array, axis=0)
        sobel_y = sobel(image_array, axis=1)
        
        # Compute the gradient magnitude
        gradient_magnitude = np.hypot(sobel_x, sobel_y)
        
        # Clip the values in the output array to be within the valid range (0 to 255)
        processed_array = np.clip(gradient_magnitude, 0, 255)
        
        # Convert the processed array back to an image
        processed_image = Image.fromarray(processed_array.astype(np.uint8))

        # Convert the image to a format Tkinter can use and display it
        tk_before = ImageTk.PhotoImage(image)
        tk_after = ImageTk.PhotoImage(processed_image)
        
        # Calculate the position to center the image on the canvas
        canvas_center_x = canvas_width // 2
        canvas_center_y = canvas_height // 2
        image_center_x = image.width // 2
        image_center_y = image.height // 2

        processed_image_center_x = image.width // 2
        processed_image_center_y = image.height // 2
        x1 = canvas_center_x - image_center_x
        y1 = canvas_center_y - image_center_y
        x2 = canvas_center_x - processed_image_center_x
        y2 = canvas_center_y - processed_image_center_y
        
        canvas1.create_image(x1, y1, anchor=tk.NW, image=tk_before)
        canvas2.create_image(x2, y2, anchor=tk.NW, image=tk_after)
        
        # Enable the canvas and bind the mouse motion event
        canvas1.bind("<Motion>", show_info)
        canvas2.bind("<Motion>", show_info)

def show_info(event):
    # Get mouse position
    x, y = event.x, event.y

    # Calculate the position to center the image on the canvas
    canvas_center_x = canvas_width // 2
    canvas_center_y = canvas_height // 2
    image_center_x = image.width // 2
    image_center_y = image.height // 2
    image_x = canvas_center_x - image_center_x
    image_y = canvas_center_y - image_center_y

    # Adjust the mouse position relative to the image
    adjusted_x = x - image_x
    adjusted_y = y - image_y

    # Check if the mouse is within the image bounds
    if 0 <= adjusted_x < image.width and 0 <= adjusted_y < image.height:
        # Get the color at the position
        color = image.getpixel((adjusted_x, adjusted_y))
        
        info_label.config(text=f"Position: ({x}, {y})\nGreyScale: {color}")
    else:
        # Clear the info label if the mouse is outside the image
        info_label.config(text="")

# Create the main Tkinter window
root = tk.Tk()
root.title("Image Intensity and Color Viewer")

# Define the fixed canvas size
canvas_width = 500
canvas_height = 600

# Create a frame for the image loading section
frame = tk.Frame(root)
frame.pack()

# Create a button to load an image
load_button = tk.Button(frame, text="Load Image", command=load_image, width=20, height=1, bg="yellow", font=("Serif", 9))
load_button.pack(side=tk.LEFT)

# Create a label to display the color and intensity information
info_label = tk.Label(root, text="", font=("Helvetica", 9))
info_label.pack(side=tk.BOTTOM)

canvas1 = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas1.pack(side=tk.LEFT, pady=5)

canvas2 = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas2.pack(side=tk.RIGHT, pady=5)

# Start the Tkinter event loop
root.mainloop()
