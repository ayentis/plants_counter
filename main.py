import cv2
import numpy as np
import os
from PIL import Image

# for en in os.environ:
#     print(en , os.environ[en])

# Image.MAX_IMAGE_PIXELS = None
# mat = Image.open(r"./source/image1.jpg")

# w, h = mat.size
# print(w, h)
# area = (1, 1, w//10, h//10)
# cropped_img = mat.crop(area)
# cropped_img.save(r"./source/cropped.jpg", format="JPEG")
# img = np.array(mat)
#
# print("end")

img = cv2.imread(r"./source/image1.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_min = np.array((36, 24, 66), np.uint8)
hsv_max = np.array((78, 255, 255), np.uint8)

mask = cv2.inRange(hsv, hsv_min, hsv_max)

# cv2.imshow('result', mask)
# ch = cv2.waitKey(5000)

# ищем контуры и складируем их в переменную contours
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
# отображаем контуры поверх изображения
cv2.drawContours(img, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)
cv2.imshow('contours', img)  # выводим итоговое изображение в окно

cv2.waitKey()
cv2.destroyAllWindows()


# imask = mask > 0
# green = np.zeros_like(img, np.uint8)
#
# green[imask] = img[imask]
#
# cv2.imwrite(r"./source/green.jpg", green)
#
print("Mask saved")

#
# print(img.shape)

# hsv_min = np.array((53, 55, 147), np.uint8)
# hsv_max = np.array((83, 160, 255), np.uint8)
#
# print(os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'])
#
# img = cv2.imread("./source/BH36_2,21ga_1ga.jpg")
#
# # преобразуем RGB картинку в HSV модель
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# # применяем цветовой фильтр
# thresh = cv2.inRange(hsv, hsv_min, hsv_max)
#
# moments = cv2.moments(thresh, 1)
# dM01 = moments['m01']
# dM10 = moments['m10']
# dArea = moments['m00']
#
# # будем реагировать только на те моменты,
# # которые содержать больше 100 пикселей
# if dArea > 100:
#     x = int(dM10 / dArea)
#     y = int(dM01 / dArea)
#     cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
#
# cv2.imshow('result', img)
#
# ch = cv2.waitKey(5)