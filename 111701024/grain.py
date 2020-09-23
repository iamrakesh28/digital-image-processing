import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

VMAX = 255
VMIN = 0
eps = 1e-6

# grain.png

# scales the intensities in [VMIN, VMAX]
def scaleImg(img):
    scaled = []
    high = img.max()
    for x in range(img.shape[0]):
        scaled.append([])
        for y in range(img.shape[1]):
            pixel = img[x][y]
            inten = int(round(pixel * (VMAX - VMIN) + VMIN))
            scaled[x].append(inten)
    return np.array(scaled)

# performs contrast stretching
def contScal(img):
    stretch = []
    H = img.max()
    L = img.min()
    img = VMAX * (img - L)
    img = img // (H - L)
    #print("L : ", L, ", H : ", H)
    return img

# performs image addition
def add(img1, img2):
    img = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.int)
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            inten = img1[x][y] + img2[x][y]
            img[x][y] = (int(inten))
    return img

# performs gamma correction
def gammaCor(img1):
    avg = (VMAX + VMIN) / 2.0 #1.0 * img.sum() / img.size
    #print(avg)
    img = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
    #print(expAvg, avg)
    # intensify black regions
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            g = img1[x][y] / avg
            inten = (1.0 * img1[x][y] / VMAX) ** g
            inten = inten * VMAX
            img[x][y] = np.uint8(inten)
    return img
    
# perfroms histogram equalisation
def histEqual(img):
    hist = []
    total = img.shape[0] * img.shape[1]
    cumFreq = [0.0 for i in range(VMAX + 1)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cumFreq[img[i][j]] += 1
        
    for val in range(VMAX + 1):
        if val:
            cumFreq[val] = cumFreq[val - 1] + 1.0 * cumFreq[val] / total
        else:
            cumFreq[val] /= total

    for x in range(img.shape[0]):
        hist.append([])
        for y in range(img.shape[1]):
            inten = img[x][y]
            hist[x].append(int(round(cumFreq[inten] * (VMAX - VMIN) + VMIN)))
                
    return np.array(hist)
    
    
def plot(img):
    imgplot = plt.imshow(img, cmap='gray', vmin=VMIN, vmax=VMAX)
    plt.show()
    
def save(img, filename):
    plt.imsave(filename+'.png', img, cmap='gray', vmin=VMIN, vmax=VMAX)

# histogram plot
def histogram(img):
    xx = [i for i in range(VMAX + 1)]
    yy = [0 for i in range(VMAX + 1)]
    for row in img:
        for col in row:
            yy[col] += 1
            #print (yy)
    plt.bar(xx, yy)
    plt.show()

# performs histogram equalisation and gamma correction and later
# adds both images with different weights

# histogram equalisation + gamma correction
if __name__ == "__main__":
    img = mpimg.imread('grain.png')
    img = scaleImg(img)
    #plot(img)
    imgHist = histEqual(img)
    imgGam = gammaCor(img)
    #histogram(imgGam)
    #plot(img1)
    #save(imgGam, 'grain-g')
    imgFinal = add(0.9 * imgHist, 0.1 * imgGam)
    plot(imgFinal)
    # final image
    save(imgFinal, 'grain-processed')
