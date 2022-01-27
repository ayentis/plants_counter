import cv2
import numpy as np
import os
from PIL import Image


def show_screen(img):
    cv2.imshow('contours', img)  # выводим итоговое изображение в окно
    cv2.waitKey()
    cv2.destroyAllWindows()


def delete_noise(img):
    kernel_size = (3, 3)  # should roughly have the size of the elements you want to remove
    kernel_el = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    show_screen(kernel_el)
    eroded = cv2.erode(img, kernel_el, (-1, -1))
    show_screen(eroded)
    cleaned_img = cv2.dilate(eroded, kernel_el, (-1, -1))

    return cleaned_img


def blur(img):
    # ksize
    ksize = (5, 5)

    # Using cv2.blur() method
    image = cv2.blur(img, ksize, cv2.BORDER_DEFAULT)
    # image = bilateralFilter(img, 9, 75, 75)
    return image


img = cv2.imread(r"./source/field1.png")
# show_screen(img)
blured_img = blur(img)
# show_screen(blured_img)

hsv = cv2.cvtColor(blured_img, cv2.COLOR_BGR2HSV)

# hsv_min = np.array((36, 24, 66), np.uint8)
# hsv_max = np.array((78, 255, 255), np.uint8)

hsv_min = np.array((26, 25, 40), np.uint8)
hsv_max = np.array((240, 220, 220), np.uint8)


mask = cv2.inRange(hsv, hsv_min, hsv_max)

show_screen(mask)

edges = cv2.Canny(mask, 50, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
result = mask.copy()
minLineLength = 100  # height/32
maxLineGap = 2  # height/40
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)

for x1, y1, x2, y2 in lines[0]:
    result = cv2.line(result, (x1, y1), (x2, y2), (255, 155, 155), 60)

# show_screen(mask)
# mask = delete_noise(mask)
# result = cv2.line(result, (50, 50), (300, 600), (255, 155, 155), 20)

show_screen(result)
# cv2.imwrite(r"./source/rez.jpg", mask)
