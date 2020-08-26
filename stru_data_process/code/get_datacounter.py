# coding: utf-8
"""
获取ROI勾勒框

"""
import cv2
import os
import matplotlib.pyplot as plt


file = "/dicom/Jone/Dection_mass/test_data/1-test/result_train_test_AI/20100901_wuqunying_1384536_RCC.png"
img = cv2.imread(file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

# plt.plot(img)
# plt.show()
cv2.imwrite("img.png", img)
exit()


def get_count(folder):
    for file in os.listdir(folder):
        file = "/dicom/Jone/Dection_mass/test_data/test1/result_AI/20100901_wuqunying_1384536_RCC.png"
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img, contours, 0, (255, 255, 255), 3)

        # plt.plot(img)
        # plt.show()
        # exit()

        # cv2.imshow(f"{folder}/{file}", img)
        cv2.imshow("img.png", img)
        exit()
        cv2.waitKey(0)

