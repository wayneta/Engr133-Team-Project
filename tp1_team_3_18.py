"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     11.2.2 tp1 team 3
    Team ID:        LC4 - 18 
    Author:         Arav Srivastava, sriva222@purdue.edu
                    Justin, 
                    Gina,
                    Ayona
    Date:           10/10/2025

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


def linearize(img_array):
    #Linearize the pixels for greyscale 
    if img_array.ndim==2:
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                c_prime=img_array[i,j]
                if c_prime<=0.04045:
                    img_array[i,j]=c_prime/12.92
                else:
                    img_array[i,j]=((c_prime+0.055)/1.055)**2.4
    
    #Linearize the pixels for color
    else:
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                for k in range(img_array.shape[2]):
                    c_prime=img_array[i,j,k]
                    if c_prime<=0.04045:
                        img_array[i,j,k]=c_prime/12.92
                    else:
                        img_array[i,j,k]=((c_prime+0.055)/1.055)**2.4
    
    return img_array

def load_img(path):
    # opening the image selected by user
    img=Image.open(path)
    # making image RGB into array
    img_array=np.array(img)
    # if statement for if the image array has 4 dimensions, then convert to 3
    if img_array.shape[-1]==4:
        img_array=img_array[:,:,:3]
    # if statement to make sure the img_array is uint8
    if img_array.dtype!=np.uint8:
        img_array=img_array.astype(np.uint8)
    # returns img_array now following our goal plans

    if img_array.all() <= 1.0:
        img_array = int(img_array.all() * 255)

    return img_array

def rgb_to_hsv(red, green, blue):

    r_prime = red / 255
    g_prime = green / 255
    b_prime = blue / 255

    C_max = max(r_prime, g_prime, b_prime)
    C_min = min(r_prime, g_prime, b_prime)
    delta = C_max - C_min

    if delta == 0:
        H_prime = 0
    elif C_max == r_prime:
        H_prime = (60 * (g_prime - b_prime) / delta) % 360
    elif C_max == g_prime:
        H_prime = ((60 * (b_prime - r_prime) / delta) + 120) % 360
    else:
        H_prime = ((60 * (r_prime - g_prime) / delta) + 240) % 360


    if C_max == 0:
        S_prime = 0
    else:
        S_prime = (delta / C_max)

    V_prime = C_max

    h = (H_prime / 360) * 255
    s = S_prime * 255
    v = V_prime * 255

    return h, s, v

def convert_to_hsv(rgb_image):
    
    rgb_array = np.array(rgb_image)
    hsv_image = np.zeros_like(rgb_array)


    height, width, _ = rgb_array.shape
    for i in range(height):
        for j in range(width):
            red, green, blue = rgb_array[i, j]
            h, s, v = rgb_to_hsv(red, green, blue)
            hsv_image[i, j] = [h, s, v]

    hsv_image = hsv_image.astype(np.uint8)

    return hsv_image


def clean_image(array):

    aspect_ratio = len(array[0]) / len(array)  # calculate width-to-height ratio
    new_width = 100  # base width for resizing
    new_height = 100  # base height for resizing

    # adjust dimensions based on aspect ratio
    if aspect_ratio < 1:  
        new_width *= aspect_ratio  # narrow image: scale width down
    elif aspect_ratio > 1:
        new_height /= aspect_ratio  # wide image: scale height down

    # convert dimensions to integers
    new_height = int(new_height)
    new_width = int(new_width)

    # resize image 
    image = Image.fromarray(array).resize(size=[new_width, new_height], resample=Image.Resampling.BILINEAR)
    print(f"Resized image to: ({new_height}, {new_width})")

    # pad image to 100×100 with black borders, centered at middle
    image = ImageOps.pad(image=image, size=[100, 100], color="black")

    # convert processed image back to numpy array
    output_array = np.array(image)

    return output_array  # return cleaned 100×100 image array

def main():
    
    image_path = input("Enter the path of the image you want to convert to hsv: ")

    image = Image.open(image_path)
    image_array = load_img(image_path)

    normalized_array = image_array / 255


    linear_array = linearize(normalized_array)

    print(type(linear_array))
    outputArray = clean_image(linear_array)

    coordinate_x, coordinate_y = map(int, input("Enter the x and y coordinates of the pixel you want to inspect: ").split(','))

    rgb_image = Image.fromarray(outputArray)

    pixel_rgb = rgb_image.getpixel((coordinate_x, coordinate_y))

    red = pixel_rgb[0]
    green = pixel_rgb[1]
    blue = pixel_rgb[2]

    print(f"RGB values of the ({coordinate_x}, {coordinate_y}) pixel: R={red}, G={green}, B={blue}")
    print(f"Converting {image_path} to HSV...")

    hsv_image = convert_to_hsv(image_array)

    image_hsv = Image.fromarray(hsv_image)
    
    pixel_hsv = image_hsv.getpixel((coordinate_x, coordinate_y))

    h = pixel_hsv[0]
    s = pixel_hsv[1]
    v = pixel_hsv[2]
    
    print(f"HSV values of the ({coordinate_x}, {coordinate_y}) pixel: H={h}, S={s}, V={v}")

    plt.imshow(hsv_image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

