import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

VMAX = 255
VMIN = 0
eps = 1e-9

class Image:
    def __init__(self, filename):
        # numpy ndarray
        self.img = mpimg.imread(filename)
        self.norm = self.img.shape[0] * self.img.shape[1]
        self.low = self.img.min()
        self.high = self.img.max()
        self.freq = None
        #print (type(self.img))

    def scaleImg(self):
        self.scaled = []
        for x in range(self.img.shape[0]):
            self.scaled.append([])
            for y in range(self.img.shape[1]):
                inten = int(round(self.img[x][y] * (VMAX - VMIN) + VMIN))
                self.scaled[x].append(inten)

    def addImg(self):
        self.add = []
        for x in range(self.img.shape[0]):
            self.add.append([])
            for y in range(self.img.shape[1]):
                inten = (self.histScaled[x][y] + self.histEqual[x][y]) // 2
                self.add[x].append(inten)
            
    def contScal(self):
        self.histScaled = []
        H = VMAX
        L = VMIN
        for inten in range(VMIN, VMAX + 1):
            if self.freq[inten] > 0:
                break
            L += 1
            
        for inten in range(VMAX, VMIN - 1, -1):
            if self.freq[inten] > 0:
                break
            H -= 1
            
        for x in range(self.img.shape[0]):
            self.histScaled.append([])
            for y in range(self.img.shape[1]):
                inten = VMAX * (self.scaled[x][y] - L)
                inten = inten // (H - L)
                self.histScaled[x].append(inten)

    def equlHist(self):
        self.histEqual = []
        cumFreq = [0.0 for i in range(VMAX + 1)]
        for val in range(VMAX + 1):
            if val:
                self.cumFreq[val] = self.cumFreq[val - 1] + 1.0 * self.freq[val] / self.norm
            else:
                self.cumFreq[val] /= self.norm
        #print(freq[VMAX])
        for x in range(self.img.shape[0]):
            self.histEqual.append([])
            for y in range(self.img.shape[1]):
                inten = self.scaled[x][y]
                self.histEqual[x].append(int(round(self.cumFreq[inten] * (VMAX - VMIN) + VMIN)))

        #print(freq)

def frequency(img):
    freq = [0 for i in range(VMAX + 1)]
    for x in range(len(img)):
        for y in range(len(img[0])):
            inten = img[x][y]
            freq[inten] += 1
    return freq
                
# Assumption: uniform histogram images are good
# square error
def cost(freq, total):
    error = 0.0
    expect = 1.0 * VMAX / total
    for inten in freq:
        error += (inten - expect) ** 2
    return error
    
    
def plot(img):
    imgplot = plt.imshow(img, cmap='gray')
    plt.show()
    
def save(img, filename):
    plt.imsave(filename+'.png', img, cmap='gray')

def histogram(img):
    xx = [i for i in range(VMAX + 1)]
    yy = [0 for i in range(VMAX + 1)]
    for row in img:
        for col in row:
            yy[col] += 1
            #print (yy)
    plt.bar(xx, yy)
    plt.show()

if __name__ == "__main__":
    img = Image('grain.png')
    img.scaleImg()
    img.freq = frequency(self.scalimg)
    print("Error (Original ) : ", cost(img.freq, img.norm))
    img.equlHist()
    print("Error (Hist. Eq.) : ", cost(frequency(img.histEqual), img.norm))
    img.histScal()
    print("Error (Con. Str.) : ", cost(frequency(img.histScaled), img.norm))
    def save(img, filename):
    plt.imsave(filename+'.png', img, cmap='gray')
