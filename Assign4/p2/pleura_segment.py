import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from queue import Queue
from skimage.segmentation import slic

# clustering based on intensity
def kmeans(img):

    # flattens the image
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    
    # define criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # number of clusters(K)
    K = 15
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # convert back into uint8
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img.shape))
    gray_res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

    # returns the segmented image
    return gray_res

dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]
visit = None

def valid(cur, size):
    if cur[0] >= 0 and cur[1] >= 0 and cur[0] < size[0] and cur[1] < size[1]:
        return True
    return False

# breadth first search for region grow
def bfs(img, src, thresh, label):
    global dr, dc, visit
    queue = Queue()

    for pos in src:
        queue.put(pos)
        visit[pos] = label
    
    while queue.qsize() > 0:
        front = queue.get()
        (r, c) = front
        for d in range(4):
            x = r + dr[d]
            y = c + dc[d]
            if valid((x, y), img.shape) == False:
                continue
            if visit[x, y] > 0:
                continue
            if abs(int(img[x, y]) - int(img[r, c])) > thresh:
                continue
            queue.put((x, y))
            visit[(x, y)] = True

    return

# finds first maximum intensity pixel
def max_indices(img, first):
    val_ind = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val_ind.append((img[i][j], (i, j)))
    val_ind.sort()
    val_ind.reverse()
    return val_ind[:first]

# region grow segmentation algorithm
def region_grow(cimg):
    global visit

    img = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    visit = np.zeros(img.shape, dtype=np.uint8)
    threshold = 3
    label = 255

    val_ind = max_indices(img, 300)
    src = []
    for val, pos in val_ind:
        src.append(pos)
    bfs(img, src, threshold, label)
    #for i in range(img.shape[0]):
    #    for j in range(img.shape[1]):
    #        if visit[i, j] == 0:
    #            bfs(img, (i, j), threshold, label)
    #            label += 1

    #plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title("Original")
    #plt.subplot(122), plt.imshow(visit, cmap='gray'), plt.title("Pleura (Region grow)")
    #plt.show()
    return visit
    

# segmentation using kmeans
def pleura_kmeans(img):

    segment_img = kmeans(img)
    pleura_img = np.array(segment_img, copy=True)
    threshold = segment_img.max()
    pleura_img[pleura_img >= threshold] = 255
    pleura_img[pleura_img < threshold] = 0

    #plt.subplot(131), plt.imshow(img), plt.title("Original")
    #plt.subplot(132), plt.imshow(segment_img, cmap='gray'), plt.title("Segmented (Kmeans)")
    #plt.subplot(133), plt.imshow(pleura_img, cmap='gray'), plt.title("Pleura Segmented")
    #plt.show()
    return pleura_img

# segemntation using plic
def super_pixels(cimg):

    segments_slic = slic(cimg, n_segments=400, compactness=10, sigma=1, start_label=1)
    img = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    # top 200 highest intensity pixels are chosen
    # to segment out the pleura region
    high_inten = max_indices(img, 200)
    pleura = np.zeros(img.shape)
    labels = set()
    for _, pos in high_inten:
        labels.add(segments_slic[pos])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if segments_slic[i, j] in labels:
                pleura[i, j] = 255
                
    #plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title("Original")
    #plt.subplot(132), plt.imshow(segments_slic, cmap='gray'), plt.title("Segmented (PLIC)")
    #plt.subplot(133), plt.imshow(pleura, cmap='gray'), plt.title("Pleura Segmented")
    #plt.show()
    return pleura
    
def main():

    input_dir = "Images/"
    filenames = os.listdir(input_dir)

    output_path = "Output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # super pixels is chosen for segmentation
    for filename in filenames:
        img = cv.imread(input_dir + filename)
        pleura = super_pixels(img)
        #pleura = region_grow(img)
        #pleura = pleura_kmeans(img)
        #plt.imshow(pleura, cmap='gray')
        #plt.show()
        plt.imsave('Output/' + filename, pleura, cmap='gray')

if __name__ == "__main__":
    main()
