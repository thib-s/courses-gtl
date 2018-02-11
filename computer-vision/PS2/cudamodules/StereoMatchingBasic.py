import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cv2
import numpy

KERNEL_CSS = """
int32_t best;

__shared__ int32_t scores[resdx];
__syncthreads();
int32_t d = 0;
int32_t tmp = 0;
for( i=-wdiam; i<=wdiam; i++){
    for( j=-wdiam; j<=wdiam; j++){
        tmp = a[(ay+i)+((ax+j)*imy)] - b[(ay+i)+((bx+j)*imy)];
        d = d + (tmp*tmp);
    }
}
scores[tix] = d;
"""

KERNEL_CROSS_CORR = """
float best;

__shared__ float scores[resdx];
__syncthreads();

int32_t aij;
int32_t bij;
float sa = 0.0;
float sb = 0.0;
float d = 0.0;
for( i=-wdiam; i<=wdiam; i++){
    for( j=-wdiam; j<=wdiam; j++){
        aij = a[(ay+i)+((ax+j)*imy)];
        bij = b[(ay+i)+((bx+j)*imy)];
        sa = sa + ((float)(aij*aij));
        sb = sb + ((float)(bij*bij));
        d = d + ((float)(aij * bij));
    }
}
scores[tix] = d / sqrtf(sa*sb);
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


def stereo_matching_basic(img1, img2, ws, kernel=KERNEL_CSS):
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
            int32_t k;
            
            """+kernel+"""
            
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
