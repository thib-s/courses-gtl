import cv2
import numpy
import pylab


def cross_corr_match(left, right, kernel, min_offset, max_offset):
    h, w = left.shape  # assume that both images are same size

    depth = numpy.zeros((w, h), numpy.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)

    for y in range(kernel_half, h - kernel_half):
        #print(".")#, end="", flush=True)  # let the user know that something is happening (slowly!)
        for x in range(kernel_half, w - kernel_half):
            if (min(w-1, x+kernel_half+max_offset)-max(0, x-kernel_half+min_offset))>kernel:
                x2 = numpy.argmax(cv2.matchTemplate(
                    right[y-kernel_half:y+kernel_half, max(0, x-kernel_half+min_offset):min(w-1, x+kernel_half+max_offset)],
                    left[y-kernel_half:y+kernel_half, x-kernel_half:x+kernel_half],
                    cv2.TM_CCOEFF_NORMED
                ))
                depth[y, x] = abs(min_offset+x2)
    return depth


if __name__ == "__main__":
    pylab.imshow(cross_corr_match(
        cv2.cvtColor(cv2.imread('Data/leftTest.png'), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread('Data/rightTest.png'), cv2.COLOR_BGR2GRAY),
        5, -2, 0), cmap=pylab.gray())
    pylab.show()
    pylab.imshow(cross_corr_match(
        cv2.cvtColor(cv2.imread('Data/proj2-pair1-L.png'), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread('Data/proj2-pair1-R.png'), cv2.COLOR_BGR2GRAY),
        10, -100, -30), cmap=pylab.gray())
    pylab.show()
