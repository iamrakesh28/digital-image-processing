import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2

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

    cA_i = y2d[0]
    cD_hor = y2d[1][0]
    cD_ver = y2d[1][1]
    cD_diag = y2d[1][2]

    img_low = reconst_col(cA_i, cD_hor, wavelet)
    img_high = reconst_col(cD_ver, cD_diag, wavelet)

    # reconstruct back the image
    cA = reconst_row(img_low, img_high, wavelet)

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

def mae(img1, img2):
    error = 0.0
    count = 0

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if i >= img2.shape[0] or j >= img2.shape[1]:
                error += img1[i][j]
            else:
                error += abs(img1[i][j] - img2[i][j])
            count += 1
            
    return error / count

def main():
    #print(pywt.wavelist('db'))

    '''
    #print(db4.dec_hi)

    # dec low
    plt.subplot(221), plt.plot(db4.dec_lo), plt.title('Decomposition Low')
    plt.subplot(222), plt.plot(db4.dec_hi), plt.title('Decomposition High')
    plt.subplot(223), plt.plot(db4.rec_lo), plt.title('Reconstruction Low')
    plt.subplot(224), plt.plot(db4.rec_hi), plt.title('Reconstruction High')
        
    plt.show()
    '''

    img = cv2.imread('../barbara.png', 0)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # print(img.shape)

    # db4 wavelet object
    # contains all the filters
    db4 = pywt.Wavelet('db4')

    
    #coeff = pywt.wavedec2(img, 'db4', level=3)
    # 3 level decompostion
    coeff = wt2d(img, db4, 3)
    
    #img_recons = pywt.waverec2(coeff, db4)
    
    img_recons = iwt2d(coeff, db4, 3)

    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(122), plt.imshow(img_recons, cmap='gray'), plt.title('Reconstructed')
    plt.show()
    
    print("Mean Absolute Error : ", mae(img, img_recons))
    
    # db4 = pywt.Wavelet('db4')
    # cA, cD = pywt.dwt([1, 2, 3, 4], 'db4')
    # print(pywt.idwt(cA, cD, 'db4'))
    # print(img.shape, cA.shape, cD.shape)
    # print(type(cA), cD)
    
if __name__ == "__main__":
    main()
