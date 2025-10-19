"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     11.2.2 tp2 team 3
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
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

def load_img(path):
    # opening the image selected by user
    img=Image.open(path)
    # making image RGB into array, and normalizes  
    img_array=np.array(img).astype(np.float64)/255.0
    #Linearlize the pixel values
    lin_array=linearize(img_array)
    # if statement for if the lineraized array has 4 dimensions, then convert to 3 (drops the alpha dimension)
    if lin_array.shape[-1]==4:
        lin_array=lin_array[:,:,:3]
    # denormalizes and returns the array 
    lin_array=(lin_array*255.0).astype(np.uint8)
    return lin_array

# UDF for linearization process. uses nested for loop to get row and column indexing
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

#UDF to convert a color image to a grayscale image
def rgb_to_grayscale(lin_array):
    # making RGB into grayscale
    gray_array=(0.2126*lin_array[:,:,0])+(0.7152*lin_array[:,:,1])+(0.0722*lin_array[:,:,2])
    return gray_array.astype(np.uint8)

def gaussian_filter(gray_array,sigma):
    #Makes sure sigma is > 0
    sigma=math.ceil(sigma)
    #Finds kernel size using given formula
    kernel_size=int(6*sigma+1)
    #Finds the size for the grid
    k=kernel_size//2
    #x values from -k to k from the center
    x=np.arange(-k,k+1)
    #y vlaues from -k to k from the center
    y=np.arange(-k,k+1)
    # relative x and y values for each position in the kernel
    xx,yy=np.meshgrid(x,y)
    #Gaussian fucntion
    gaus_xy=(1/(2*np.pi*sigma**2))*(np.e**(-(((xx**2)+(yy**2))/(2*sigma**2))))
    #Normalize kernel
    kernel=gaus_xy/np.sum(gaus_xy)
    #gets the height and width of the array
    height,width=np.shape(gray_array)
    #creates a zero array same size as gray array
    blurred=np.zeros_like(gray_array)
    #pads the edges of gray array
    padded=np.pad(gray_array,k,mode='edge')
    #loop to blur each pixel
    #loops through all the rows
    for y in range(height):
        #loops through all the columns
        for x in range(width):
            #finds the 3x3 region surounding the pixel
            region=padded[y:y+kernel_size,x:x+kernel_size]
            #blurred value for each pixel
            blurred[y,x]=np.sum(region*kernel)
    #returns blurred array
    return blurred.astype(np.uint8)

def detect_circles(gray_img):
 """Detects if a large circle is present. Returns 1 if found, 0 otherwise."""
 # Hough Circles works best on a grayscale, slightly blurred image
 # Apply a Gaussian blur to reduce noise
 blurred_img = np.array(gaussian_filter(gray_img, sigma=1.5))

 # Detect circles
 circles = cv2.HoughCircles(
     blurred_img,
     cv2.HOUGH_GRADIENT,
     dp=1.2,  # Inverse ratio of accumulator resolution
     minDist=100,  # Minimum distance between centers of detected circles
     param1=150,  # Upper threshold for the internal Canny edge detector
     param2=50,  # Threshold for center detection
     minRadius=20,  # Minimum circle radius to detect
     maxRadius=50  # Maximum circle radius to detect

 )

 # Return 1 if any circles are found, otherwise 0
 return 1 if circles is not None else 0

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

def count_lines(gray_img):
 """Counts the number of lines in an image using Hough Line Transform."""
 # Edge detection via sobel filtering is a prerequisite for Hough Lines
 edges = sobel_filter(gray_img)

 # Detect lines using the edge map
 lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

 # Return the number of lines found
 return len(lines) if lines is not None else 0


def convert_to_hsv(rgb_image):

    rgb_array = np.array(rgb_image)

    hsv_image = np.zeros_like(rgb_array)

    height, width, _ = rgb_array.shape
    for i in range(height):
        for j in range(width):
            red, green, blue = rgb_array[i, j]
            h, s, v = rgb_to_hsv(red, green, blue)
            hsv_image[i, j] = [h, s, v]

    hsv_array = hsv_image.astype(np.uint8)

    return hsv_array

def extract_features(clean_rgb_img):

    hsv_array = convert_to_hsv(clean_rgb_img)
    rgb_array = np.array(clean_rgb_img)

    hue_channel = hsv_array[:, :, 0]
    value_channel = hsv_array[:, :, 1]
    sat_channel = hsv_array[:, :, 2]

    hue_mean = np.average(hue_channel)
    saturation_mean = np.average(sat_channel)
    value_mean = np.average(value_channel)
    

    hue_std = np.std(hue_channel)
    saturation_std = np.std(sat_channel)
    value_std = np.std(value_channel)


    grayscale_array = rgb_to_grayscale(rgb_array)
    gray_img = Image.fromarray(grayscale_array)

    has_circles = detect_circles(gray_img)
    number_lines = count_lines(gray_img)

    return  hue_mean, value_mean, saturation_mean, hue_std, value_std, saturation_std, has_circles, number_lines



def main():

    type shit = 123
    
if __name__ == "__main__":
    main()
