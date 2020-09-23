import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = mpimg.imread('../../Pictures/Aster-Alexxis_crop.jpg')
print (len(img), len(img[0]), len(img[0][0]))
imgplot = plt.imshow(img)

grey = []
for i in range(len(img)):
    grey.append([])
    for j in range(len(img[i])):
        grey[i].append(sum(img[i][j]) / len(img[i][j]))

greyplot = plt.imshow(grey)
plt.show()
