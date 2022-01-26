import cv2
import numpy as np
import os
from PIL import Image


def show_screen(img):
    cv2.imshow('contours', img)  # выводим итоговое изображение в окно
    cv2.waitKey()
    cv2.destroyAllWindows()


def delete_noise(img):
    kernel_size = (5, 5)  # should roughly have the size of the elements you want to remove
    kernel_el = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded = cv2.erode(img, kernel_el, (-1, -1))
    cleaned_img = cv2.dilate(eroded, kernel_el, (-1, -1))

    return cleaned_img


def blur(img):
    # ksize
    ksize = (15, 15)

    # Using cv2.blur() method
    image = cv2.blur(img, ksize, cv2.BORDER_DEFAULT)
    return image


img = cv2.imread(r"./source/cabbage.jpeg")
blured_img = blur(img)

hsv = cv2.cvtColor(blured_img, cv2.COLOR_BGR2HSV)

hsv_min = np.array((36, 24, 66), np.uint8)
hsv_max = np.array((78, 255, 255), np.uint8)

mask = cv2.inRange(hsv, hsv_min, hsv_max)

cleaned_mask = delete_noise(mask)

# ищем контуры и складируем их в переменную contours
# contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(cleaned_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print(f"Image contain {len(contours)} contours")

# отображаем контуры поверх изображения
cv2.drawContours(img, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)
show_screen(img)
cv2.imwrite(r"./source/rez.jpg", img)


