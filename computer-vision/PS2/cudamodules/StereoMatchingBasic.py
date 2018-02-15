import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cv2
import numpy
import pylab

KERNEL_CSS = """
register int64_t best;

__shared__ int64_t scores[resdy];
__syncthreads();
register int64_t d = 0;
register int64_t tmp = 0;
for( i=-wdiam; i<=wdiam; i++){
    for( j=-wdiam; j<=wdiam; j++){
        tmp = a[(ay+i)+((ax+j)*imy)] - b[(bx+i)+((ax+j)*imy)];
        d = d + (tmp*tmp);
    }
}
scores[tix] = d;
"""

KERNEL_CROSS_CORR = """
register float best;

__shared__ float scores[resdx];
__syncthreads();

int64_t aij;
int64_t bij;
int64_t sa = 0;
int64_t sb = 0;
float d = 0.0;
float div = 0.0;
// split the computation in two steps to avoid overflow
for( i=-wdiam; i<=wdiam; i++){
    for( j=-wdiam; j<=wdiam; j++){
        aij = a[(ay+i)+((ax+j)*imy)];
        bij = b[(bx+i)+((ax+j)*imy)];
        sa = sa + (aij*aij);
        sb = sb + (bij*bij);
    }
}
div = sqrtf((float)sa)*sqrtf((float)sb);
for( i=-wdiam; i<=wdiam; i++){
    for( j=-wdiam; j<=wdiam; j++){
        aij = a[(ay+i)+((ax+j)*imy)];
        bij = b[(bx+i)+((ax+j)*imy)];
        d  = d  + ((aij*bij) / div);
    }
}
scores[tix] =  d;
"""


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


def stereo_matching_basic(img1, img2, ws, kernel=KERNEL_CSS, dmin=30, dmax=120):
    (imx, imy) = img1.shape
    assert (img2.shape == (imx, imy))
    # block x and block y => x and y position of the mask on image a
    # thread x => x position of the mask on image b
    mod = SourceModule("""
          __global__ void compute(int64_t *a, int64_t *b, int64_t *result)
          {
            const int64_t bix = blockIdx.x;
            const int64_t biy = blockIdx.y;
            const int64_t tix = threadIdx.x;
            const int64_t ws = """ + str(ws) + """;
            const int64_t imx = """ + str(imx) + """;
            const int64_t imy = """ + str(imy) + """;
            const int64_t wdiam = (ws-1)/2;
            const int64_t resdx = imx-ws+1;
            const int64_t resdy = imy-ws+1;
            const int64_t ax = (bix + wdiam);
            const int64_t ay = (biy + wdiam);
            const int64_t bx = tix + wdiam;
            register int64_t i;
            register int64_t j;
            register int64_t k;
            
            """ + kernel + """
            
            __syncthreads();
            if (tix == 0){
            """+ str('for (k=0; k<resdx; k++){' if (dmin >= 0) else 'for (k=(resdx-1); k>=0; k--){')
                +"""
                    d = scores[k];
                    if ((k==0)||(d <= best)){
                        best = d;
                        if (((biy - k)>=""" + str(dmin) + """)&&((biy - k)<=""" + str(dmax) + """)){ 
                            result[biy + (bix*resdy)] = biy - k;
                        }
                    }
                }
            }
          }
          """)
    func = mod.get_function("compute")

    img1 = img1.astype(numpy.int64)
    img2 = img2.astype(numpy.int64)
    img1_gpu = cuda.mem_alloc(img1.nbytes)
    img2_gpu = cuda.mem_alloc(img2.nbytes)
    cuda.memcpy_htod(img1_gpu, img1)
    cuda.memcpy_htod(img2_gpu, img2)

    matches = numpy.full((imx - ws + 1, imy - ws + 1), min(abs(dmax), abs(dmin)), dtype=numpy.int64)
    matches_gpu = cuda.mem_alloc(matches.nbytes)
    cuda.memcpy_htod(matches_gpu, matches)

    func(img1_gpu, img2_gpu, matches_gpu, block=(imx - ws + 1, 1, 1),
         grid=(imx - ws + 1, imy - ws + 1))
    cuda.memcpy_dtoh(matches, matches_gpu)
    if dmin < 0:
        matches = matches * -1
    return matches


if __name__ == "__main__":
    pylab.imshow(stereo_matching_basic(
        cv2.cvtColor(cv2.imread('Data/leftTest.png'), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread('Data/rightTest.png'), cv2.COLOR_BGR2GRAY),
        5), cmap=pylab.gray())
    pylab.show()

    pylab.imshow(stereo_matching_basic(
        cv2.cvtColor(cv2.imread('Data/leftTest.png'), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread('Data/rightTest.png'), cv2.COLOR_BGR2GRAY),
        5, kernel=KERNEL_CROSS_CORR), cmap=pylab.gray())
    pylab.show()

    pylab.imshow(stereo_matching_basic(
        cv2.cvtColor(cv2.imread("Data/proj2-pair1-L.png"), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread("Data/proj2-pair1-R.png"), cv2.COLOR_BGR2GRAY),
        10), cmap=pylab.gray())
    pylab.show()

    pylab.imshow(stereo_matching_basic(
        cv2.cvtColor(cv2.imread("Data/proj2-pair1-L.png"), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread("Data/proj2-pair1-R.png"), cv2.COLOR_BGR2GRAY),
        5, kernel=KERNEL_CROSS_CORR), cmap=pylab.gray())
    pylab.show()
