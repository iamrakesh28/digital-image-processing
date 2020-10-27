import numpy as np
import matplotlib.pyplot as plt
import pywt

if __name__ == "__main__":
    #print(pywt.wavelist('db'))
    # db4 filters
    db4 = pywt.Wavelet('db4')
    #print(db4.dec_hi)

    # dec low
    plt.subplot(221), plt.plot(db4.dec_lo), plt.title('Decomposition Low')
    plt.subplot(222), plt.plot(db4.dec_hi), plt.title('Decomposition High')
    plt.subplot(223), plt.plot(db4.rec_lo), plt.title('Reconstruction Low')
    plt.subplot(224), plt.plot(db4.rec_hi), plt.title('Reconstruction High')
        
    plt.show()
