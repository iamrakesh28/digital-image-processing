import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def valid(x, y, m, n):
    if x >= 0 and y >= 0 and x < m and y < n:
        return True
    return False

def conv(img, kernel):
    newImg = []
    #odd
    kx = (kernel.shape[0] - 1) // 2
    ky = (kernel.shape[1] - 1) // 2
    for x in range(img.shape[0]):
        newImg.append([])
        for y in range(img.shape[1]):
            con = 0
            for rx in range(-kx, kx + 1):
                for ry in range(-ky, ky + 1):
                    if valid(x + rx, y + ry, img.shape[0], img.shape[1]):
                        con += img[x + rx][y + ry] * kernel[rx][ry]
            #print(con)
            newImg[x].append(max(0, con))
    return np.array(newImg)

def plot(img):
    imgplot = plt.imshow(img, cmap='gray')
    plt.show()
    
img = mpimg.imread('quiz3.png')
kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
kernel2 = [[-1, 0, 0], [0, 0, 0], [0, 0, 1]]
print(kernel)
img1 = conv(img, np.array(kernel2))
plot(img1)
kernel = [[-2, -2, 0], [-2, 0, 2], [0, 2, 2]]
#img2 = conv(img, np.array(kernel))
#plot(img2)
