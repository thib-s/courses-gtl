import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cv2
import numpy


def testCuda():
    a = numpy.random.randn(4, 4)

    a = a.astype(numpy.float32)

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
          __global__ void compute(int *a, int *b, int *dists, int *result)
          {
            const int ws = 9;
            const imx = 512;
            const imy = 512;
            const wdiam = (ws-1)/2;
            int ax = (blockIdx.x + wdiam)
            int ay = (blockIdx.y + wdiam);
            int bx = (threadIdx.x+ wdiam);
            int i;
            int j;
            int d=0;
            int tmp;
            for( i=-wdiam; i<=wdiam; i++){
                for( j=-wdiam; j<=wdiam; j++){
                    tmp = a[(ax+i)+((ay+j)*imx)] - a[(bx+i)+((ay+j)*imx)];
                    d += tmp**2;
                }
            }
            if ((dists[ax+(ay*imx)]==-1) || (d<=dists[ax+(ay*imx)])){
                result[ax+(ay*imx)] = bx;
            }
          }
          """)
    pass
