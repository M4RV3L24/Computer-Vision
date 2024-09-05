        for row in range(image_height):
            for col in range(image_width):
                # Apply the kernel to the surrounding pixels
                rgb = np.zeros(3)
                for rdif in range(-offX, offX+1):
                    for cdif in range(-offY, offY+1):
                        if (row + rdif) >= image_height or (row + rdif) < 0 or (col + cdif) >= image_width or (col + cdif) < 0:
                            continue
                        for i in range(0, 3):
                            rgb[i] += image_array[row + rdif, col + cdif, i] * kernel[rdif + offX, cdif + offY]
                
                # Assign the result to the corresponding pixel in the output array
                processed_array[row, col] = rgb
