# Critical Thinking 4
# Britney Clark
# CSC515: Foundations fo Computer Vision
# Dr. Jonathan Issa
# February 13, 2022

import cv2
import numpy as np

# Read image

img = cv2.imread(r'C:\Users\Stardust\Downloads\Mod4CT1.jpg')

# Mean Filtering AKA Average Filter
mean_im1 = cv2.blur(img, (3, 3))
mean_im2 = cv2.blur(img, (5, 5))
mean_im3 = cv2.blur(img, (7, 7))

new_mean_im1 = cv2.putText(mean_im1, 'Mean Filter 3x3', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
new_mean_im2 = cv2.putText(mean_im2, 'Mean Filter 5x5', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
new_mean_im3 = cv2.putText(mean_im3, 'Mean Filter 7x7', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

# Median Filtering
med_im1 = cv2.medianBlur(img, 3)
med_im2 = cv2.medianBlur(img, 5)
med_im3 = cv2.medianBlur(img, 7)

new_med_im1 = cv2.putText(med_im1, 'Median Filter 3x3', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
new_med_im2 = cv2.putText(med_im2, 'Median Filter 5x5', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
new_med_im3 = cv2.putText(med_im3, 'Median Filter 7x7', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

# Gaussian Filtering
gaus_im11 = cv2.GaussianBlur(img, (3, 3), 1)
gaus_im12 = cv2.GaussianBlur(img, (3, 3), 4)
gaus_im21 = cv2.GaussianBlur(img, (5, 5), 1)
gaus_im22 = cv2.GaussianBlur(img, (5, 5), 4)
gaus_im31 = cv2.GaussianBlur(img, (7, 7), 1)
gaus_im32 = cv2.GaussianBlur(img, (7, 7), 4)

new_gaus_im11 = cv2.putText(gaus_im11, 'Gauss Filter 3x3, sigma = 1', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
new_gaus_im12 = cv2.putText(gaus_im12, 'Gauss Filter 3x3, sigma = 4', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
new_gaus_im21 = cv2.putText(gaus_im21, 'Gauss Filter 5x5, sigma = 1', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
new_gaus_im22 = cv2.putText(gaus_im22, 'Gauss Filter 5x5, sigma = 4', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
new_gaus_im31 = cv2.putText(gaus_im31, 'Gauss Filter 7x7, sigma = 1', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
new_gaus_im32 = cv2.putText(gaus_im32, 'Gauss Filter 7x7, sigma = 4', (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

colm1 = np.concatenate((new_mean_im1, new_mean_im2, new_mean_im3), axis=0)
colm2 = np.concatenate((new_med_im1, new_med_im2, new_med_im3), axis=0)
colm3 = np.concatenate((new_gaus_im11, new_gaus_im21, new_gaus_im31), axis=0)
colm4 = np.concatenate((new_gaus_im12, new_gaus_im22, new_gaus_im32), axis=0)

cv2.imshow('Filtered Images', np.hstack((colm1, colm2, colm3, colm4)))
cv2.waitKey(0)
