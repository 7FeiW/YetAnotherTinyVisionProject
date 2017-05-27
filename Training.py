# import the necessary packages
import cv2
import os
import numpy as np

def get_hog(image_path):
    # load the image and resize it
    image = cv2.imread(image_path, 0)
    # setup hogs
    n_bins = 36
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(n_bins * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), n_bins) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist


def fill_train_data(sub_dir, class_sign):
    train_list = []
    response_list = []
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), sub_dir).replace("\\","/")):
        image_path = os.path.join(sub_dir, filename)
        hist = get_hog(image_path).tolist()
        train_list.append(hist)
        response_list.append(class_sign)
    return train_list, response_list

train_list = []
response_list = []

sub_dir = 'data/Singles/Duke1'
train_data_temp, responses_temp = fill_train_data(sub_dir, 0)
train_list += train_data_temp
response_list += responses_temp

sub_dir = 'data/Singles/Duke2'
train_data_temp, responses_temp = fill_train_data(sub_dir, 1)
train_list += train_data_temp
response_list += responses_temp

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)

svm.setC(1.0)
train_mat = np.array(train_list, np.float32)
response_mat = np.array(response_list)[np.newaxis].T

#training
svm.train(train_mat, cv2.ml.ROW_SAMPLE, response_mat)
svm.save('svm_data.dat')

hog_hist = np.array(get_hog('data/Singles/Duke1/Duke1-1.png'), np.float32)[np.newaxis]
print(svm.predict(hog_hist))
hog_hist = np.array(get_hog('data/Singles/Duke2/Duke2-1.png'), np.float32)[np.newaxis]
print(svm.predict(hog_hist))