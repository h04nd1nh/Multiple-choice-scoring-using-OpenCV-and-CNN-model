import imutils
import numpy as np
import cv2
from math import ceil
from model import CNN_Model
import matplotlib.pyplot as plt
import imutils
#  hàm nhận đầu vào list, mỗi phần tử trong list có dạng [image, (x, y, w, h)] và trả về tọa độ x:
def get_x(s):
    return s[1][0]

#  hàm nhận đầu vào list, mỗi phần tử trong list có dạng [image, (x, y, w, h)] và trả về tọa độ y:
def get_y(s):
    return s[1][1]

#  hàm nhận đầu vào list, mỗi phần tử trong list có dạng [image, (x, y, w, h)] và trả về chiều cao h:
def get_h(s):
    return s[1][3]

# hàm nhận đầu vào là một contour và trả về diện tích của hình chữ nhật bao quanh contour đó :
def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def crop_image(img):

    # resize_img = cv2.resize(img, 1056,1500)
    # # convert image from BGR to GRAY to apply canny edge detection algorithm
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise by blur image
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # apply canny edge detection algorithm
    img_canny = cv2.Canny(blurred, 100, 200)

    # find contours
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    

    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, cnts, -1, (0, 255, 0), 2)

    # Hiển thị hình ảnh gốc và hình ảnh với contours
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)), plt.title('Image with Contours')
    plt.show()

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in descending order
        cnts = sorted(cnts, key=get_x_ver1)

        # loop over the sorted contours
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

            if w_curr * h_curr > 100000:
                # check overlap contours
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                # if list answer box is empty
                if len(ans_blocks) == 0:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr

        # sort ans_blocks according to x coordinate
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        return sorted_ans_blocks



def process_ans_blocks(ans_blocks):
    """
        this function process 2 block answer box and return a list answer has len of 200 bubble choices
        :param ans_blocks: a list which include 2 element, each element has the format of [image, [x, y, w, h]]
    """
    list_answers = []

    # Loop over each block ans in
    for ans_block in ans_blocks:
        ans_block_img = np.array(ans_block[0])

        offset1 = ceil(ans_block_img.shape[0] / 6)
        # Loop over each box in answer block
        for i in range(6):
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            height_box = box_img.shape[0]

            box_img = box_img[14:height_box - 14, :]
            offset2 = ceil(box_img.shape[0] / 5)

            # loop over each line in a box
            for j in range(5):
                list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])

    return list_answers



img1 = cv2.imread('test1.jpg')


crop_image(img1)

