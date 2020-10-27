import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot(img, filt_img):
    # rows -> 1, cols -> 2, index -> 1
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(filt_img, cmap='gray'), plt.title('Median Filtering')
    plt.xticks([]), plt.yticks([])

    plt.show()

# N = 13 -> image_1.JPG
# N = 9  -> image_2.JPG
# filter size should such that it covers mimima
# as well as maxima of one cycle

# don't make size large otherwise features will be lost
if __name__ == "__main__":
    #img = cv2.imread('../image_1.JPG')
    img = cv2.imread('../image_2.JPG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape)

    #filt_img = cv2.medianBlur(img, 13)
    filt_img = cv2.medianBlur(img, 9)
    plot(img, filt_img)

    plt.imsave('median_2.jpg', filt_img, cmap='gray')
