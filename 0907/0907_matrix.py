import numpy as np

A = np.array ( [[1,1], [0,1]] )
B = np.array( [[2,0], [3,4]] )
C = A*B # 각 element 곱한거
D = A@ B # 행렬연산
X = [1,2,3,4]
Y = [1,0,1,0]
F = np.inner(X,Y) # 내적? (1+3 = 4)?

print(A)
print("=========")
print(B)
print("=========")
print(C)
print("=========")
print(D)
print("=========")
print(F)