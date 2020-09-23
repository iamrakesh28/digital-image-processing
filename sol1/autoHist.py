import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# VMAX -> white
# VMIN -> black
VMAX = 255
VMIN = 0
eps = 1e-6

# scales the image intensities in the range [VMIN, VMAX] (integer)
# should be used when intensities in [0, 1]

def scaleImg(img):
    scaled = []
    high = img.max()
    for x in range(img.shape[0]):
        scaled.append([])
        for y in range(img.shape[1]):
            pixel = img[x][y]
            # checks whether the image has more than 1 channel
            if len(img.shape) > 2:
                pixel = 1.0 * sum(pixel) / img.shape[2]
            if high < 1 + eps:
                inten = int(round(pixel * (VMAX - VMIN) + VMIN))
            else:
                inten = int(pixel)
            scaled[x].append(inten)
    return np.array(scaled)

# does contrast stetching
def contScal(img):
    stretch = []
    H = img.max()
    L = img.min()
    img = VMAX * (img - L)
    img = img // (H - L)
    #print("L : ", L, ", H : ", H)
    return img

# performs gamma correction
# assumes the intensites to be in [VMIN, VMAX]
# returns the image with intensities in [0, 1] (float)
def gammaCor(img):
    expAvg = (VMAX + VMIN) / 2.0
    avg = 1.0 * img.sum() / img.size

    #gamma is average intensity / 127.5
    g = avg / expAvg
    #print(expAvg, avg)
    img = img / (1.0 * VMAX)
    img = img ** g
    return img
    
#performs histogram equalisation
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
    imgplot = plt.imshow(img, cmap='gray')
    plt.show()

# saves the image in grayscale
def save(img, filename):
    plt.imsave(filename+'.png', img, cmap='gray')


# histogram plot
# image intensities should be in [VMIN, VMAX]
def histogram(img):
    xx = [i for i in range(VMAX + 1)]
    yy = [0 for i in range(VMAX + 1)]
    for row in img:
        for col in row:
            yy[col] += 1
            #print (yy)
    plt.bar(xx, yy)
    plt.show()

# performs contrast stretching -> histogram equalisation -> gamma correction
if __name__ == "__main__":
    # input image
    img = mpimg.imread('lena.jpg')
    img = scaleImg(img)
    #plot(img)
    #histogram(img)
    #save(img, 'original')
    img = contScal(img)
    #save(img, 'con-stretch')
    #plot(img)
    #histogram(img)
    img = histEqual(img)
    #plot(img)
    #histogram(img)
    #save(img, 'hist-equal')
    img = gammaCor(img)
    # final image
    save(img, 'processed')
