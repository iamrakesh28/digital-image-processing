import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# rotates the kirsch kernel to get the next kernel
def kirsch_rot(kernel):
    temp = kernel[0][0]
    kernel[0][0] = kernel[0][1]
    kernel[0][1] = kernel[0][2]
    kernel[0][2] = kernel[1][2]

    kernel[1][2] = kernel[2][2]

    kernel[2][2] = kernel[2][1]
    kernel[2][1] = kernel[2][0]

    kernel[2][0] = kernel[1][0]
    kernel[1][0] = temp
    
    return kernel

# checks whether (x, y) is inside the image or not
def valid(x, y, r, c):
    if x >= 0 and y >= 0 and x < r and y < c:
        return True
    return False

# performs 2D convolution on the image
def filter2D(img, kernel):
    output = np.zeros(img.shape)
    for r_img in range(img.shape[0]):
        for c_img in range(img.shape[1]):
            r_range = kernel.shape[0] // 2
            c_range = kernel.shape[1] // 2
            conv = 0
            for r_ker in range(kernel.shape[0]):
                for c_ker in range(kernel.shape[1]):
                    r_patch = r_ker + r_img - r_range
                    c_patch = c_ker + c_img - c_range
                    dot = 0
                    if valid(r_patch, c_patch, img.shape[0], img.shape[1]):
                        dot = img[r_patch][c_patch] * kernel[r_ker][c_ker]
                    conv += dot
            output[r_img][c_img] = conv
    return output

# detects edges using kirsch kernel
def kirsch_edge(cimg):
    
    img = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    
    base_kernel = [[5, 5, 5],
                   [-3, 0, -3],
                   [-3, -3, -3]]
    edges = None
    for i in range(8):
        base_kernel = kirsch_rot(base_kernel)
        conv = cv.filter2D(img, -1, np.array(base_kernel))
        if i == 0:
            edges = conv
        else:
            edges = np.maximum(edges, conv)
            
    ret, edges = cv.threshold(edges, 150, 255, cv.THRESH_BINARY)
    return edges

# performs laplacian of gaussian on the image
def marr_edge(cimg):
    img = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    
    gaussian = cv.GaussianBlur(img, (3, 3), 0)

    # ksize == 1 is filtering the image with [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplacian = cv.Laplacian(gaussian, None, 1, ksize=1)

    #lap_filter = [[0, 1, 0],
    #              [0, -4, 0],
    #              [0, 1, 0]]
    #laplacian = filter2D(gaussian, np.array(lap_filter))
    return laplacian
    
# performs hough circle transform on the image
def hough_circle(img_edge, img_org):

    img_org = np.array(img_org, copy=True)
    # Open CV Hough Circles
    # img :  Input image (grayscale).
    # HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
    # dp = 1: The inverse ratio of resolution.
    # min_dist : Minimum distance between detected centers.
    # param1 : Upper threshold for the internal Canny edge detector.
    # param2 : Threshold for center detection.
    # minRadius : Minimum radius to be detected. If unknown, put zero as default.
    # maxRadius : Maximum radius to be detected. If unknown, put zero as default.
    # circles: A vector that stores sets of 3 values: xc, yc, r for each detected circle
    
    circles = cv.HoughCircles(img_edge, cv.HOUGH_GRADIENT, 1, 50,
                              param1=90, param2=10, minRadius=10, maxRadius=30)
    
    circles = np.uint16(np.around(circles))
    # print(circles)

    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(img_org, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        # cv.circle(img_org,(i[0], i[1]), 2, (0, 0, 255), 3)

    return img_org

def subplots(img, h_img, text):
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title(text)
    plt.subplot(122), plt.imshow(h_img)
    plt.title("Hough transform")
    plt.imsave(text + "_Hough.png", h_img)
    plt.show()
    
def main():
    img = cv.imread("coins.png", cv.IMREAD_COLOR)
    edge_kirsch = kirsch_edge(img)
    hough_kirsch = hough_circle(edge_kirsch, img)
    subplots(edge_kirsch, hough_kirsch, "Kirsch filter")

    # setting lower threshold results in unwanted circles
    # lower threshold -> 245
    # upper threshold -> 255
    edge_canny = cv.Canny(img, 245, 255)
    hough_canny = hough_circle(edge_canny, img)
    subplots(edge_canny, hough_canny, "Canny Edge Detection")
    plt.imsave("boundaries.png", hough_canny)

    edge_marr = marr_edge(img)
    hough_marr = hough_circle(edge_marr, img)
    subplots(edge_marr, hough_marr, "Marr-Hildreth")

if __name__ == "__main__":
    main()
