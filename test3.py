import imutils
import numpy as np
import cv2
from math import ceil
from model import CNN_Model
import matplotlib.pyplot as plt
import imutils

def get_x(s):
    return s[1][0]

# Function to get the y-coordinate from a list where each element is of the form [image, (x, y, w, h)]
def get_y(s):
    return s[1][1]

# Function to get the height (h) from a list where each element is of the form [image, (x, y, w, h)]
def get_h(s):
    return s[1][3]

# Function to get the x-coordinate using boundingRect from a contour
def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]

img = cv2.imread('yen.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove noise by blurring the image
blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Apply Canny edge detection algorithm
img_canny = cv2.Canny(blurred, 100, 200)

# Find contours
cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
