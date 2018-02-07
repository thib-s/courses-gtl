import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cv2
import numpy
import pylab


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


def stereo_matching_basic(img1, img2):
    # block x and block y => x and y position of the mask on image a
    # thread x => x position of the mask on image b
    mod = SourceModule("""
          __global__ void compute(int32_t *a, int32_t *b, int32_t *result)
          {
            const int32_t ws = 9;
            const int32_t imx = 128;
            const int32_t imy = 128;
            const int32_t wdiam = (ws-1)/2;
            int32_t ax = (blockIdx.x + wdiam);
            int32_t ay = (blockIdx.y + wdiam);
            int32_t bx = (threadIdx.x+ wdiam);
            int32_t i;
            int32_t j;
            int32_t d = 0;
            int32_t tmp;
            
            __shared__ int32_t scores[imx-ws+1];
            scores[blockIdx.x] = 0;
            __syncthreads();
            
            for( i=-wdiam; i<=wdiam; i++){
                for( j=-wdiam; j<=wdiam; j++){
                    tmp = a[(ax+i)+((ay+j)*imx)] - b[(bx+i)+((ay+j)*imx)];
                    d = d + (tmp*tmp);
                }
            }
            if ((scores[blockIdx.x]==(-1)) || (d<=scores[blockIdx.x])){
                scores[blockIdx.x] = d;
                result[blockIdx.x+(blockIdx.y*(imx-ws+1))] = ax-bx;
            }
          }
          """)
    func = mod.get_function("compute")

    ws = 9
    (imx, imy) = img1.shape
    assert (img2.shape == (imx, imy))

    img1 = img1.astype(numpy.int32)
    img2 = img2.astype(numpy.int32)
    img1_gpu = cuda.mem_alloc(img1.nbytes)
    img2_gpu = cuda.mem_alloc(img2.nbytes)
    cuda.memcpy_htod(img1_gpu, img1)
    cuda.memcpy_htod(img2_gpu, img2)

    matches = numpy.zeros((imx - ws+1, imy - ws+1), dtype=numpy.int32)
    matches_gpu = cuda.mem_alloc(matches.nbytes)
    cuda.memcpy_htod(matches_gpu, matches)

    dists = numpy.full((imx - ws+1, imy - ws+1), 100000000, dtype=numpy.int32)
    dists_gpu = cuda.mem_alloc(dists.nbytes)
    cuda.memcpy_htod(dists_gpu, dists)
    func(img1_gpu, img2_gpu, matches_gpu, block=(imx - ws + 1, 1, 1),
         grid=(imx - ws + 1, imy - ws + 1))
    cuda.memcpy_dtoh(matches, matches_gpu)
    return matches


img1 = cv2.imread('Data/leftTest.png')
img1 = img1[:, :, 1]
img2 = cv2.imread('Data/rightTest.png')
img2 = img2[:, :, 1]
# pylab.imshow(img1, cmap=pylab.gray())
# pylab.show()
# pylab.imshow(img2, cmap=pylab.gray())
# pylab.show()

print(img2.shape)
print(numpy.min(img2))
print(numpy.max(img2))
matches = stereo_matching_basic(img1, img2)
pylab.imshow(matches, cmap=pylab.gray())
pylab.show()
print(matches.shape)
print(numpy.min(matches))
print(numpy.max(matches))
