import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cv2
import numpy
import pylab

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def testCuda():
    a = numpy.random.randn(4, 4)

    a = a.astype(numpy.float64)

    a_gpu = cuda.mem_alloc(a.nbytes)

    cuda.memcpy_htod(a_gpu, a)

    mod = SourceModule("""
      __global__ void doublify(float *a)
      {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
      }
      """)

    func = mod.get_function("doublify")
    func(a_gpu, block=(4, 4, 1))

    a_doubled = numpy.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)
    print(a_doubled)
    print(a)


def stereo_matching_basic(img1, img2, ws):
    (imx, imy) = img1.shape
    assert (img2.shape == (imx, imy))
    # block x and block y => x and y position of the mask on image a
    # thread x => x position of the mask on image b
    mod = SourceModule("""
          __global__ void compute(int32_t *a, int32_t *b, int32_t *result)
          {
            const int32_t bix = blockIdx.x;
            const int32_t biy = blockIdx.y;
            const int32_t tix = threadIdx.x;
            const int32_t ws = """+str(ws)+""";
            const int32_t imx = """+str(imx)+""";
            const int32_t imy = """+str(imy)+""";
            const int32_t wdiam = (ws-1)/2;
            int32_t ax = (blockIdx.x + wdiam);
            const int32_t resdx = imx-ws+1;
            const int32_t resdy = imy-ws+1;
            int32_t ay = (blockIdx.y + wdiam);
            int32_t bx = threadIdx.x + wdiam;
            int32_t i;
            int32_t j;
            int32_t d = 0;
            int32_t tmp;
            int32_t k;
            int32_t best;
            
            __shared__ int32_t scores[resdx];
            __syncthreads();
            
            for( i=-wdiam; i<=wdiam; i++){
                for( j=-wdiam; j<=wdiam; j++){
                    tmp = a[(ay+i)+((ax+j)*imy)] - b[(ay+i)+((bx+j)*imy)];
                    d = d + (tmp*tmp);
                }
            }
            scores[tix] = d;
            
            __syncthreads();
            if (tix == 0){
                for (k=0; k<resdx; k++){
                    d = scores[k];
                    if ((k==0)||(d < best)){
                        best = d;
                        result[biy + (bix*resdy)] = bix - k;
                    }
                }
            }
          }
          """)
    func = mod.get_function("compute")

    img1 = img1.astype(numpy.int32)
    img2 = img2.astype(numpy.int32)
    img1_gpu = cuda.mem_alloc(img1.nbytes)
    img2_gpu = cuda.mem_alloc(img2.nbytes)
    cuda.memcpy_htod(img1_gpu, img1)
    cuda.memcpy_htod(img2_gpu, img2)

    matches = numpy.zeros((imx - ws+1, imy - ws+1), dtype=numpy.int32)
    matches_gpu = cuda.mem_alloc(matches.nbytes)
    cuda.memcpy_htod(matches_gpu, matches)

    func(img1_gpu, img2_gpu, matches_gpu, block=(imx - ws + 1, 1, 1),
         grid=(imx - ws + 1, imy - ws + 1))
    cuda.memcpy_dtoh(matches, matches_gpu)
    return matches


# img1 = cv2.imread('Data/leftTest.png')
img1 = cv2.imread('Data/proj2-pair1-L.png')
# img1 = cv2.GaussianBlur( src=img1, ksize=(7,7), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REFLECT)
img1 = img1[:, :, 1]
# img2 = cv2.imread('Data/rightTest.png')
img2 = cv2.imread('Data/proj2-pair1-R.png')
# img2 = cv2.GaussianBlur( src=img2, ksize=(7,7), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REFLECT)
img2 = img2[:, :, 1]
# pylab.imshow(img1, cmap=pylab.gray())
# pylab.show()
# pylab.imshow(img2, cmap=pylab.gray())
# pylab.show()

print(img2.shape)
print(numpy.min(img2))
print(numpy.max(img2))
ws = 9
matches = stereo_matching_basic(img1, img2, ws)
pylab.imshow(matches, cmap=pylab.gray())
pylab.show()
print(matches.shape)
print(numpy.min(matches))
print(numpy.max(matches))


# Set up grid and test data
(imy, imx) = matches.shape
x = range(imx)
y = range(imy)


hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, matches)

plt.show()
