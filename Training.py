# import the necessary packages
import datetime
import imutils
import cv2
import numpy as np

# load the image and resize it
image = cv2.imread('./data/images/print-3_1.png', 0)

# setup hogs
win_size = (64,64)
block_size = (16,16)
block_stride = (8,8)
cell_size = (8,8)
n_bins = 9
deriv_aperture = 1
win_sigma = 4.
histogram_norm_type = 0
L2_hys_threshold = 2.0000000000000001e-01
gamma_correction = 0
n_levels = 64

# compute hog
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
image = imutils.resize(image, width=min(400, image.shape[1]))

win_stride = (0, 0)
padding = (0, 0)
hist = hog.compute(image, win_stride, padding)
print(len(hist))
# show the output image
cv2.imshow("Detections", image)
cv2.waitKey(0)