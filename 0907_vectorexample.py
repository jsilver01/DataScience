import time
import numpy as np
import random

def numpy_create():
    start = time.time()
    # 1000 이하의 정수를 백만 개 생성해서 NumPy 배열에 저장
    a = np.random.randint(1, 1001, 1000000)
    
    # 배열에서 500보다 큰 수는 1로, 500 이하는 0으로 변환
    new_a = np.where(a > 500, 1, 0)
    
    end = time.time()
    print("NumPy elapsed time =", end - start)

def list_create():
    start = time.time()
    num_list = [random.randint(1, 1000) for _ in range(1000000)]
    
    # 리스트에서 500보다 큰 수는 1로, 500 이하는 0으로 변환
    new_list = [1 if num > 500 else 0 for num in num_list]
    
    end = time.time()
    print("List elapsed time =", end - start)

if __name__ == "__main__":
    numpy_create()
    list_create()
