import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2

# global list for approximation reconstructions
img_recons = []

# reconstruct the rows
def reconst_row(cA, cD, wavelet):
    n_rows = min(cA.shape[0], cD.shape[0])

    img_rec = []

    for row in range(n_rows):
        min_len = min(cA[row].shape[0], cD[row].shape[0])
        # 1 D reconstruction
        img_row = pywt.idwt(cA[row][:min_len], cD[row][:min_len], wavelet)
        img_rec.append(img_row)

    return np.array(img_rec)


# reconstruct the cols
def reconst_col(cA, cD, wavelet):
    cA = cA.T
    cD = cD.T
    n_rows = min(cA.shape[0], cD.shape[0])

    img_rec = []

    for row in range(n_rows):
        min_len = min(cA[row].shape[0], cD[row].shape[0])
        # 1 D recnstruction
        img_row = pywt.idwt(cA[row][:min_len], cD[row][:min_len], wavelet)
        img_rec.append(img_row)

    cA = cA.T
    cD = cD.T
    
    return np.array(img_rec).T

# performs the reconstruction
# y2d -> [cA, (cD_hor_n, cD_ver_n, cD_diag_n), (..)]
def iwt2d(y2d, wavelet, nlevels):

    global img_recons
    
    # gets the 4 coefficients
    cA_i = y2d[0]
    cD_hor = y2d[1][0]
    cD_ver = y2d[1][1]
    cD_diag = y2d[1][2]

    img_low = reconst_col(cA_i, cD_hor, wavelet)
    img_high = reconst_col(cD_ver, cD_diag, wavelet)

    # reconstruct back the image
    cA = reconst_row(img_low, img_high, wavelet)
    img_recons.append(cA)

    if nlevels > 1:
        return iwt2d([cA] + y2d[2:], wavelet, nlevels - 1)
    else:
        return cA

# decomposes the rows of the image x2d
def decompose_row(x2d, wavelet):
    n_rows = x2d.shape[0]

    c_low = []
    c_high = []
    for row in range(n_rows):
        # 1 D decompostion
        cA, cD = pywt.dwt(x2d[row], wavelet)
        c_low.append(cA)
        c_high.append(cD)

    c_low = np.array(c_low)
    c_high = np.array(c_high)

    return (c_low, c_high)

# decomposes the cols of the image x2d
def decompose_col(x2d, wavelet):

    x2d = x2d.T
    n_rows = x2d.shape[0]

    c_low = []
    c_high = []
    for row in range(n_rows):
        # 1 D decompostion
        cA, cD = pywt.dwt(x2d[row], wavelet)
        c_low.append(cA)
        c_high.append(cD)

    x2d = x2d.T
    c_low = np.array(c_low)
    c_high = np.array(c_high)

    return (c_low.T, c_high.T)
    
# wavelet object will have all the filters
def wt2d(x2d, wavelet, nlevels):
    
    n_rows = x2d.shape[0]
    n_cols = x2d.shape[1]

    cA, cD = decompose_row(x2d, wavelet)
    cA, cD_hor = decompose_col(cA, wavelet)
    cD_ver, cD_diag = decompose_col(cD, wavelet)

    final = []

    if nlevels > 1:
        final = wt2d(cA, wavelet, nlevels - 1)
        final.append((cD_hor, cD_ver, cD_diag))
    else:
        final = [cA, (cD_hor, cD_ver, cD_diag)]

    return final

def main():
    #print(pywt.wavelist('db'))

    img = cv2.imread('../barbara.png', 0)
    #plt.imshow(img, cmap='gray')
    #plt.show()
    #print(img.shape)

    # db4 wavelet object
    # it has all the filters
    db4 = pywt.Wavelet('db4')

    
    #coeff = pywt.wavedec2(img, 'db4', level=2)
    # 3 level decompostion
    coeff = wt2d(img, db4, 3)

    for i in range(1, len(coeff)):
        # initialize detail coefficient to zeros
        tup = (np.zeros((coeff[i][0].shape), dtype=np.float32),
               np.zeros((coeff[i][1].shape), dtype=np.float32),
               np.zeros((coeff[i][2].shape), dtype=np.float32))

        coeff[i] = tup

    #img_recons = pywt.waverec2(coeff, db4)
    # reconstructed image
    img_rec = iwt2d(coeff, db4, 3)

    plt.subplot(232), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(234), plt.imshow(img_recons[0], cmap='gray'), plt.title('Reconstructed Level 3')
    plt.subplot(235), plt.imshow(img_recons[1], cmap='gray'), plt.title('Reconstructed Level 2')
    plt.subplot(236), plt.imshow(img_recons[2], cmap='gray'), plt.title('Reconstructed Level 1')
    plt.show()

    
if __name__ == "__main__":
    main()
