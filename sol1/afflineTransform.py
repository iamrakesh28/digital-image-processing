import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.interpolate import griddata

VMAX = 255
VMIN = 0
eps = 1e-6
inf = int(1e9)
valid = set()

# scales the image intensities in [VMIN, VMAX]
def scaleImg(img):
    scaled = []
    high = img.max()
    for x in range(img.shape[0]):
        scaled.append([])
        for y in range(img.shape[1]):
            pixel = img[x][y]
            # if the image has more than 1 channel
            if len(img.shape) > 2:
                pixel = 1.0 * sum(pixel) / img.shape[2]
            if high < 1 + eps:
                inten = int(round(pixel * (VMAX - VMIN) + VMIN))
            else:
                inten = int(pixel)
            scaled[x].append(inten)
    return np.array(scaled)

# performs affline transformation
# img -> image
# T -> transformation matrix
# meth -> interpolation method
# inv -> it used when we do inverse transformation of the image.
#        As, after the original transformation the image may not
#        be a rectangle. To make it a rectangle, some extra places
#        filled with numpy.nan (Not a Number). So, it is important
#        to considered those pixels while performing inverse.
#        inv == True -> we are taking inverse
# valid  -> stores the valid image points that should be only used
#        finding the inverse
def affline(img, T, meth, inv=False):
    global valid
    # H and V stores the maximum and minumum x and y coordinates
    H = [inf, -inf]
    V = [inf, -inf]
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if inv:
                if (x, y) not in valid:
                    continue
            point = (int(T[0][0] * x +  T[0][1] * y + T[0][3]),
                     int(T[1][0] * x +  T[1][1] * y + T[1][3]))
            H[0] = min(H[0], point[1])
            H[1] = max(H[1], point[1])
            V[0] = min(V[0], point[0])
            V[1] = max(V[1], point[0])
    # new image size
    b = H[1] - H[0] + 1
    l = V[1] - V[0] + 1
    # used for interpolation
    points = []
    values = []
    # mesh grid for image
    gridX, gridY = np.mgrid[0:l, 0:b]
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if inv:
                if (x, y) not in valid:
                    continue
            point = (int(T[0][0] * x +  T[0][1] * y + T[0][3]),
                     int(T[1][0] * x +  T[1][1] * y + T[1][3]))
            xx = point[0] - V[0]
            yy = point[1] - H[0]
            if inv == False:
                valid.add((xx, yy))
            # points and their values
            points.append([xx, yy])
            values.append(img[x][y])

    # using the points and their values, griddata returns the interpolated image
    return griddata(np.array(points), np.array(values), (gridX, gridY), method=meth)
    #save(img, 'transformed')
    
def plot(img, title):
    plt.title(title)
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
    img = mpimg.imread('lena.jpg')
    img = scaleImg(img)
    # original image
    plot(img, "Original")
    tx = 0
    ty = 0
    # T -> transformation matrix
    T = np.array([[0.3, 0.1, 0, tx], [0.5, 1.9, 0, ty], [0, 0, 1, 0]])
    inv = np.linalg.inv(T[:, :3])
    # Tinv -> inverse transformation matrix
    Tinv = np.zeros((3, 4))
    for i in range(3):
        for j in range(3):
            Tinv[i][j] = inv[i][j]
    T[0][3] = -tx
    T[1][3] = -ty
    img = affline(img, T, "linear")
    # transformed image
    plot(img, "Transformed")
    # Do not try to save without replacing the nan values
    #save(img, "transform")
    # inverse image
    img = affline(img, Tinv, "linear", True)
    plot(img, "Inverse")
    #save(img, "revert")
