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

    def scaleImg(self):
        self.scaled = []
        for x in range(self.img.shape[0]):
            self.scaled.append([])
            for y in range(self.img.shape[1]):
                pixel = self.img[x][y]
                if len(self.img.shape) > 2:
                    pixel = 1.0 * sum(pixel) / self.img.shape[2]
                if self.high < 1 + eps:
                    inten = int(round(pixel * (VMAX - VMIN) + VMIN))
                else:
                    inten = int(pixel)
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
        delta = 1.0 * self.norm / (VMAX - VMIN)
        sum = 0.0
        expect = 0.0
        for inten in range(VMIN, VMAX + 1):
            sum += self.freq[inten]
            expect += delta
            if sum > expect / 6:
                break
            L += 1

        sum = 0.0
        expect = 0.0
        for inten in range(VMAX, VMIN - 1, -1):
            sum += self.freq[inten]
            expect += delta
            if sum > expect / 6:
                break
            H -= 1
        #H = 140
        #L = 80
        print("L : ", L, ", H : ", H)
        for x in range(self.img.shape[0]):
            self.histScaled.append([])
            for y in range(self.img.shape[1]):
                if self.scaled[x][y] < L:
                    self.histScaled[x].append(0)
                    continue
                if self.scaled[x][y] > H:
                    self.histScaled[x].append(VMAX)
                    continue
                inten = VMAX * (self.scaled[x][y] - L)
                inten = inten // (H - L)
                self.histScaled[x].append(inten)

    def equlHist(self):
        self.histEqual = []
        cumFreq = [0.0 for i in range(VMAX + 1)]
        for val in range(VMAX + 1):
            if val:
                cumFreq[val] = cumFreq[val - 1] + 1.0 * self.freq[val] / self.norm
            else:
                cumFreq[val] /= self.norm
        #print(freq[VMAX])
        for x in range(self.img.shape[0]):
            self.histEqual.append([])
            for y in range(self.img.shape[1]):
                inten = self.scaled[x][y]
                self.histEqual[x].append(int(round(cumFreq[inten] * (VMAX - VMIN) + VMIN)))

        #print(freq)

def frequency(img):
    freq = [0 for i in range(VMAX + 1)]
    for x in range(len(img)):
        for y in range(len(img[x])):
            inten = img[x][y]
            #print(inten)
            freq[inten] += 1
    return freq
    
    
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
    img = Image('lena.jpg')
    img.scaleImg()
    img.freq = frequency(img.scaled)
    #histogram(img.scaled)
    #histogram(img.histEqual)
    img.contScal()
    img.freq = frequency(img.histScaled)
    img.equlHist()
    save(img.scaled, 'original')
    save(img.histEqual, 'hist-equal')
    save(img.histScaled, 'con-stretch')
