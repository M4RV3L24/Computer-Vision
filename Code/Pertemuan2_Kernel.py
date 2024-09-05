import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import ImageOps
import numpy as np

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
        image_array = np.array(image)

        # make kernel for Gaussian Blur filter
        sobel_kernelX = np.array(
            [[-1, 0, 1], 
             [-2, 0, 2], 
             [-1, 0, 1]]
             )
        
        sobel_kernelY = np.array(
            [[-1, -2, -1], 
             [0, 0, 0], 
             [1, 2, 1]]
             )
        
        

        
        # kernel = kernel/kernel.sum()
        


        # Get the dimensions of the image and kernel
        image_height = image_array.shape [0]
        image_width = image_array.shape [1]
        kernel_height, kernel_width = sobel_kernelX.shape

        # apply filter to image
        processed_array = np.copy(image_array)
        
        # Define the offset for the kernel (assuming the kernel is square and has odd dimensions)
        offX = kernel_height // 2
        offY = kernel_width // 2

        # Iterate over each pixel in the image (excluding the borders)
        # FOR RGB
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

        
        # Clip the values in the output array to be within the valid range (0 to 255)
        processed_array = np.clip(processed_array, 0, 255)
        np.set_printoptions(threshold=np.inf)
        # print(processed_array)

        
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

    # print("Intensity At (0, 0)",image.getpixel((0, 0)))


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
