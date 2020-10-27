import cv2
import numpy as np
import matplotlib.pyplot as plt

# calculates PSNR
def psnr(max_i, mean_se):
    psnr_db = 10 * np.log10(max_i * max_i / mean_se)
    return psnr_db

# Calculates MSE
def mse(img_org, img_noisy):
    sq_error = (img_org - img_noisy) ** 2
    m, n = sq_error.shape
    mean_sq_err = sq_error.sum() / (m * n)

    return mean_sq_err 
    
def main():
    img = cv2.imread('../barbara.png', 0)
    # print (img.shape)
    # (mean, sigma, shape)
    gauss_noise = np.random.normal(0, 10, img.shape)
    noisy_img = img + gauss_noise

    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(122), plt.imshow(noisy_img, cmap='gray'), plt.title('Noisy')
    plt.show()

    # MSE
    mean_se = mse(img, noisy_img)

    # psnr in dB
    # maximum pixel possible value here is 255
    psnr_db = psnr(255, mean_se)
    print("PSNR (in dB) : ", psnr_db)
    
if __name__ == "__main__":
    main()
