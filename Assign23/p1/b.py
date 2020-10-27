import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot(img, dft_img):
    # rows -> 1, cols -> 2, index -> 1
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dft_img, cmap='gray'), plt.title('Magnitude Spectrum')
    plt.xticks([]), plt.yticks([])

    plt.show()

# image_1 -> the noise is going in x direction
#            so, it has mainly u component

# image_2 -> the noise is both u and v factor
if __name__ == "__main__":
    img = cv2.imread('../image_2.JPG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape)

    # 2D DFT
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)
    #print(dft.shape)

    # magnitude
    # log to scale down the value
    magn_spect = 20 * np.log(np.abs(dft))
    plot(img, magn_spect)

    plt.imsave('dft_2.jpg', magn_spect, cmap='gray')
