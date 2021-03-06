import matplotlib.pyplot as plt
import matplotlib.image as mpimg

VMAX = 255
VMIN = 0

class Image:
    def __init__(self, filename):
        # numpy ndarray
        self.img = mpimg.imread(filename)
        self.norm = self.img.shape[0] * self.img.shape[1]
        self.low = self.img.min()
        self.high = self.img.max()
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
            
    # H = 70
    # L = 0
    # MAX = 110
    def histScal(self):
        self.histScaled = []
        H = 80
        L = 0
        #MAX = 110
        MAX = 180
        for x in range(self.img.shape[0]):
            self.histScaled.append([])
            for y in range(self.img.shape[1]):
                if (self.scaled[x][y] >= H):
                    #inten = VMAX * (self.scaled[x][y] - H)
                    #inten = inten // (VMAX - H)
                    self.histScaled[x].append(max(MAX, self.scaled[x][y]))
                    continue
                inten = MAX * (self.scaled[x][y] - L)
                inten = inten // (H - L)
                self.histScaled[x].append(self.scaled[x][y] + MAX)

    def equlHist(self):
        freq = [0.0 for i in range(VMAX + 1)]
        self.histEqual = []
        for x in range(self.img.shape[0]):
            self.histEqual.append([])
            for y in range(self.img.shape[1]):
                self.histEqual[x].append(0)
                inten = self.scaled[x][y]
                freq[inten] += 1.0

        for val in range(VMAX + 1):
            if val:
                freq[val] = freq[val - 1] + freq[val] / self.norm
            else:
                freq[val] /= self.norm
        #print(freq[VMAX])
        for x in range(self.img.shape[0]):
            for y in range(self.img.shape[1]):
                inten = self.scaled[x][y]
                self.histEqual[x][y] = int(round(freq[inten] * (VMAX - VMIN) + VMIN))

        #print(freq)
        

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
    #print(img.low, img.high)
    img.equlHist()
    img.histScal()
    img.addImg()
    #histogram(img.scaled)
    #histogram(img.histEqual)
    #histogram(img.histScaled)
    #histogram(img.add)
    #plot(img.scaled)
    #plot(img.histEqual)
    #plot(img.histScaled)
    #plot(img.add)

    save(img.scaled, "original")
    save(img.histEqual, "histogram-equalized")
    save(img.histScaled, "contrast-stretched")
    save(img.add, "addition")
