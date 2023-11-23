import time
import numpy as np

def list_comp():
    a = [r for r in range(10000000)]
    b = [r for r in range(10000000)]
    c = []
    
    start = time.time()
    for i in range(10000000): c.append(a[i]*b[i])
    end = time.time()

    print("elasped time =", end-start)

def ndarray_comp():
    a = np.arange(10000000) 
    b = np.arange(10000000)

    start = time.time()
    c = a * b
    end = time.time()

    print("elasped time =", end-start)

if __name__ == "__main__":
    list_comp()
    ndarray_comp()