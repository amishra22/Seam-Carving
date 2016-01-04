from matplotlib.pyplot import *
import numpy
from skimage import *
from skimage.filters import sobel_h, sobel_v
from numpy import *


def dual_gradient_energy(img1):
    w, h = img1.shape[:2]
    R = img1[:, :, 0]
    G = img1[:, :, 1]
    B = img1[:, :, 2]
    A = sobel_h(R)**2 + sobel_v(R)**2 + \
        sobel_h(G)**2 + sobel_v(G)**2 +\
        sobel_h(B)**2 + sobel_v(B)**2
    r = len(range(len(A)))
    c = len(range(len(A[0])))
    for i in range(len(A[0])):
        A[0][i] = A[1][i]
        A[r - 1][i] = A[r - 2][i]
    for i in range(len(A)):
        A[i][0] = A[i][1]
        A[i][c - 1] = A[i][c - 2]
    return A


def get_min(x, y, index):
    if x > y:
        return y, index + 1
    else:
        return x, index


def find_seam(img1):
    H = len(img1)
    W = (len(img1[0]))
    cost_matrix = numpy.zeros([H, W])
    I = numpy.zeros([H, W], dtype=int)

    for i in range(H):
        for j in range(W):
            if i is 0:
                cost_matrix[i][j] = img1[i][j]
                I[i][j] = 0
            else:
                cost_matrix[i][j] = sys.maxint
                I[i][j] = -1

    for x in range(1, H):
        for y in range(W):
            if y is 0:
                min_val, index = \
                    get_min(cost_matrix[x - 1][y], cost_matrix[x-1][y+1], y)
            elif y < len(img1[0]) - 1:
                min_val1, index1 = \
                    get_min(cost_matrix[x - 1][y], cost_matrix[x-1][y+1], y)
                min_val2, index2 = \
                    get_min(min_val1, cost_matrix[x-1][y-1], y-1)
                if min_val1 == min_val2:
                    min_val = min_val1
                    index = index1
                else:
                    min_val = min_val2
                    index = y - 1
            else:
                min_val, index = \
                    get_min(cost_matrix[x - 1][y - 1],
                            cost_matrix[x - 1][y], y - 1)
            cost_matrix[x][y] = img1[x][y] + min_val
            I[x][y] = index

    min = sys.maxint
    index_min = -1

    for j in range(0, W):
        sum = cost_matrix[H - 1][j]
        if sum < min:
            min = sum
            index_min = j
    print "Min cost of the seam removed", min

    seam = zeros(H, dtype=int32)
    seam[H - 1] = index_min
    for i in range(H - 2, -1, -1):
        seam[i] = I[i + 1][seam[i + 1]]
    seam = insert(seam, 0, seam[0])
    seam = append(seam, seam[H - 1])
    return seam


def remove_seam(img, seam):
    img = img_as_float(img)
    img = img.tolist()
    seam = seam.tolist()
    for i in range(0, len(img)):
        del img[i][seam[i]]
    matplotlib.pyplot.imshow(img)
    return img


def plot_seam(img, seam):
    A = dual_gradient_energy(img)
    s = []
    for i in range(0, len(A)):
        s.append((seam[i], i))
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.plot(*zip(*s), color='r')
    matplotlib.pyplot.imshow(A)


def remove_multiple_seam(img, number):
    for i in range(0, number):
        img1 = dual_gradient_energy(img)
        seam = find_seam(img1)
        print "Removing seam: ", i + 1
        img = remove_seam(img, seam)
        img = img_as_float(img)
        matplotlib.pyplot.tight_layout()
        subplot(1, 5, 4)
        plot_seam(img, seam)
        title("50 seams")
    return img


if __name__ == '__main__':
    img = imread("test.png")
    figure()
    gray()
    img = img_as_float(img)
    img1 = dual_gradient_energy(img)
    print "Removing just a single seam"
    subplot(1, 5, 1)
    imshow(img)
    title("Original image")
    seam = find_seam(img1)
    subplot(1, 5, 2)
    plot_seam(img, seam)
    title("1 Seam")
    subplot(1, 5, 3)
    remove_seam(img, seam)
    title("1 Seam removed")
    print "\n\n........\n Now removing 50 seams in a loop\n.......\n"
    img = remove_multiple_seam(img, 50)
    subplot(1, 5, 5)
    matplotlib.pyplot.imshow(img)
    title("50 seams removed")
    show()
    matplotlib.pyplot.imsave("carved.png", img)
