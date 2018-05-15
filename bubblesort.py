from timeit import default_timer as timer

import numpy

from numba import jit,cuda


def bubblesort(X):
    N = X.shape[0]
    for end in range(N, 1, -1):
        for i in range(end - 1):
            if X[i] > X[i + 1]:
                X[i], X[i + 1] = X[i + 1], X[i]


bubblesort_fast = cuda.jit()(bubblesort)

dtype = numpy.int64

def main():
    N = 5000
    blockSize = 256
    numBlock = int((N+blockSize -1)/blockSize)
    arr = numpy.array(list(reversed(range(N))),dtype=dtype)

    print('== Test CPU ==')
    t0 = timer()
    X0 = arr.copy()
    bubblesort(X0)
    print(X0)
    cpu_time = timer() - t0

    print('== Test GPU == ')
    t1 = timer()
    X1 = arr.copy()
    bubblesort_fast[numBlock,blockSize](X1)
    bubblesort_fast(X1)
    print(X1)
    gpu_time = timer() - t1

    print('CPU', cpu_time)
    print('GPU', gpu_time)



if __name__ == '__main__':
    main()
