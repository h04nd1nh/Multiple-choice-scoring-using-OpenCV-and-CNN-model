import imutils
import numpy as np
import cv2
from math import ceil
from model import CNN_Model
from collections import defaultdict


def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def crop_image(img):
    # convert image from BGR to GRAY to apply canny edge detection algorithm
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise by blur image
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # apply canny edge detection algorithm
    img_canny = cv2.Canny(blurred, 100, 200)

    # find contours
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

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
    
    list_answers = []

   
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


def process_list_ans(list_answers):
    list_choices = []
    offset = 44
    start = 32

    for answer_img in list_answers:
        for i in range(4):
            bubble_choice = answer_img[:, start + i * offset:start + (i + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)

    if len(list_choices) != 480:
        raise ValueError("Length of list_choices must be 480")
    return list_choices


def map_answer(idx):
    if idx % 4 == 0:
        answer_circle = "A"
    elif idx % 4 == 1:
        answer_circle = "B"
    elif idx % 4 == 2:
        answer_circle = "C"
    else:
        answer_circle = "D"
    return answer_circle


def get_answers(list_answers):
    results = defaultdict(list)
    model = CNN_Model('weight.h5').build_model(rt=True)
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        question = idx // 4

        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # choiced confidence score > 0.9
            chosed_answer = map_answer(idx)
            results[question + 1].append(chosed_answer)

    return results



def get_answers_from_exam(image_path):
    img = cv2.imread(image_path)
    list_ans_boxes = crop_image(img)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    answers = get_answers(list_ans)
    return answers


if __name__ == '__main__':
    img = cv2.imread('test1.jpg')
    list_ans_boxes = crop_image(img)
    # list_ans = process_ans_blocks(list_ans_boxes)
    # list_ans = process_list_ans(list_ans)
    # answers = get_answers(list_ans)
    print("Answer:")
    print(list_ans_boxes)
